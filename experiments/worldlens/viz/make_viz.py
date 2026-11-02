"""Side-by-side visualisation of SUV rollouts (RGB / depth / seg / instance) against GT.

CPU only (numpy + PIL + matplotlib + system ffmpeg); no torch import, no GPU.

    PYTHONPATH=$REPO python -m experiments.worldlens.viz.make_viz \
        --raw-root /data-work/SUV/worldlens/rollouts/wl-a-navtest1k-s0/raw \
        --out /outputs/SUV/runs/wl-a-navtest1k-s0/viz            # 24-scene default selection
    ... --tokens 0000548db87959c2 0027991369e05ab2               # arbitrary scenes

Frame alignment (from ``suv_rollout.py`` + ``navsimv1.pdm_agent._build_history_condition_video``
with ``history_padding_mode='repeat_first'``; also checked numerically, see ``check_alignment``):

    gt_rgb   [12] = h0 h1 h2 h3 | f1 .. f8          h3 = current frame t0 (t-1.5 s .. t0, t+0.5 .. t+4.0 s)
    pred_*   [13] = h0 h0 h1 h2 h3 | f1 .. f8       index 0 is the repeated pad frame
    => time k*0.5 s (k = -3..8):  gt index k+3,  pred index k+4.

Outputs under ``--out``::

    mp4/<token>.mp4      context (t-1.5 s) .. t+4.0 s, each 2 Hz step held ``--sec-per-step`` seconds
    grid/<token>.png     rows = GT RGB, SUV RGB, depth, seg, instance; cols = t+0.5 .. t+4.0 s
    thumbs/<token>.jpg   index thumbnail (GT | SUV at t+2.0 s)
    selection.json       chosen tokens + selection rule + per-scene ADE / command
    index.html           thumbnail grid with mp4 / grid links
"""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.worldlens import layout

DT = 0.5  # seconds per step (2 Hz)
COMMANDS = ("turn left", "go straight", "turn right", "follow the lane")  # navsimv1/data/prompts.py
DIFF_VMAX = 128.0  # |GT-SUV| colour scale (mean abs over RGB, 0..255)

BG = (18, 18, 20)
FG = (235, 235, 235)
CTX_COLOR = (90, 140, 230)
FUT_COLOR = (235, 120, 60)


# --------------------------------------------------------------------------- alignment
def frame_indices(meta: dict) -> List[Tuple[int, int, int]]:
    """[(k, gt_idx, pred_idx)] for k = -(n_hist-1) .. n_fut, time = k * DT."""
    n_hist = int(meta["num_history_frames"])
    n_fut = int(meta["num_future_frames"])
    n_cond = int(meta["num_condition_frames"])
    pad = n_cond - n_hist  # repeat_first -> 1, none -> 0
    if meta.get("history_padding_mode", "repeat_first") == "repeat_first":
        assert pad == 1, meta
    elif meta.get("history_padding_mode") == "none":
        assert pad == 0, meta
    out = []
    for k in range(-(n_hist - 1), n_fut + 1):
        out.append((k, k + n_hist - 1, k + n_hist - 1 + pad))
    return out


def check_alignment(arrays: Dict[str, np.ndarray], meta: dict) -> dict:
    """Pixel MAE of decoded pred frames vs GT; the claimed pairing must be the row-wise argmin
    for the condition frames (decoded context is a near-reconstruction of the GT history)."""
    g = arrays["gt_rgb"].astype(np.float32)
    p = arrays["pred_rgb"].astype(np.float32)
    n_hist = int(meta["num_history_frames"])
    ok = True
    maes = []
    for k, gi, pi in frame_indices(meta):
        if k > 0:
            break
        row = np.abs(p[pi][None] - g).mean(axis=(1, 2, 3))
        maes.append(float(row[gi]))
        ok &= int(np.argmin(row)) == gi
    return {"ok": bool(ok), "context_mae": maes, "n_hist": n_hist}


# --------------------------------------------------------------------------- drawing helpers
_FONT_CACHE: Dict[int, ImageFont.FreeTypeFont] = {}


def font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _FONT_CACHE:
        try:
            from matplotlib import font_manager

            path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
            _FONT_CACHE[size] = ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def command_name(cmd) -> str:
    v = np.asarray(cmd, dtype=np.float32).reshape(-1)
    if v.size == 0 or np.abs(v).sum() < 1e-6:
        return COMMANDS[3]
    i = int(np.argmax(v))
    return COMMANDS[i] if i < len(COMMANDS) else COMMANDS[3]


def resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img.shape[1] == w and img.shape[0] == h:
        return img
    return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))


def label(img: np.ndarray, text: str, size: int = 16, color=FG, corner: str = "tl") -> np.ndarray:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    f = font(size)
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=f)
    tw, th = x1 - x0, y1 - y0
    pad = 4
    if corner == "tl":
        x, y = 0, 0
    elif corner == "tr":
        x, y = im.width - tw - 2 * pad, 0
    elif corner == "bl":
        x, y = 0, im.height - th - 2 * pad
    else:
        x, y = im.width - tw - 2 * pad, im.height - th - 2 * pad
    d.rectangle([x, y, x + tw + 2 * pad, y + th + 2 * pad], fill=(0, 0, 0))
    d.text((x + pad - x0, y + pad - y0), text, font=f, fill=color)
    return np.asarray(im)


_INFERNO = None


def diff_map(gt: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, float]:
    global _INFERNO
    if _INFERNO is None:
        from matplotlib import colormaps

        _INFERNO = (colormaps["inferno"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    d = np.abs(gt.astype(np.int16) - pred.astype(np.int16)).mean(axis=-1)
    idx = np.clip(d / DIFF_VMAX * 255.0, 0, 255).astype(np.uint8)
    return _INFERNO[idx], float(d.mean())


def bev_panel(meta: dict, w: int, h: int, step: Optional[int]) -> np.ndarray:
    """Top-down ego-frame trajectories. NAVSIM ego frame: x forward, y left -> plot (-y, x).
    ``step`` = number of future waypoints reached (0..8) or None for all."""
    import matplotlib.pyplot as plt

    pred = np.asarray(meta["pred_trajectory"], dtype=np.float32)[:, :2]
    gt = np.asarray(meta["gt_trajectory"], dtype=np.float32)[:, :2]
    hist = np.asarray(meta.get("ego_status_history", np.zeros((0, 3))), dtype=np.float32)
    hist = hist[:, :2] if hist.size else np.zeros((0, 2), np.float32)

    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor=np.array(BG) / 255)
    ax = fig.add_axes([0.14, 0.09, 0.82, 0.76])
    ax.set_facecolor((0.10, 0.10, 0.12))
    pts = np.concatenate([pred, gt, hist, np.zeros((1, 2))], axis=0)
    lat = -pts[:, 1]
    lon = pts[:, 0]
    # equal metric scale: x span / y span = axes pixel width / height
    r = (0.82 * w) / (0.76 * h)
    sy = max(lon.max() - lon.min(), (lat.max() - lat.min()) / r, 10.0) * 1.15
    sx = sy * r
    cx, cy = (lat.max() + lat.min()) / 2, (lon.max() + lon.min()) / 2
    ax.set_xlim(cx - sx / 2, cx + sx / 2)
    ax.set_ylim(cy - sy / 2, cy + sy / 2)
    if len(hist):
        ax.plot(-hist[:, 1], hist[:, 0], "-o", color="0.6", ms=3, lw=1.2, label="past")
    g = np.concatenate([np.zeros((1, 2)), gt])
    p = np.concatenate([np.zeros((1, 2)), pred])
    ax.plot(-g[:, 1], g[:, 0], "-o", color="#4fc36a", ms=3.5, lw=1.8, label="GT")
    ax.plot(-p[:, 1], p[:, 0], "-o", color="#f07a3c", ms=3.5, lw=1.8, label="SUV")
    if step is not None and step > 0:
        ax.plot(-gt[step - 1, 1], gt[step - 1, 0], "o", ms=10, mfc="none", mec="#4fc36a", mew=2)
        ax.plot(-pred[step - 1, 1], pred[step - 1, 0], "o", ms=10, mfc="none", mec="#f07a3c", mew=2)
    ax.plot([0], [0], marker="^", color="white", ms=9)
    ax.grid(True, color="0.3", lw=0.5)
    ax.tick_params(colors="0.8", labelsize=8)
    for s in ax.spines.values():
        s.set_color("0.4")
    ax.set_xlabel("lateral [m] (right +)", color="0.8", fontsize=8)
    ax.set_ylabel("forward [m]", color="0.8", fontsize=8)
    ax.legend(loc="lower right", fontsize=8, facecolor=(0.15, 0.15, 0.17), edgecolor="0.4", labelcolor="0.9")
    fig.text(0.03, 0.955, "BEV trajectory (ego @ t0)", color="white", fontsize=10, weight="bold", va="top")
    fig.text(0.03, 0.905, f"cmd: {command_name(meta['driving_command'])}   ADE: {meta['ade_m']:.2f} m",
             color="#ffd27f", fontsize=9, va="top")
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return resize(img, w, h)


# --------------------------------------------------------------------------- per-scene renderers
def compose_frame(arrays, meta, k: int, gi: int, pi: int, pw: int, ph: int, bev: np.ndarray) -> np.ndarray:
    gt = arrays["gt_rgb"][gi]
    prgb = arrays["pred_rgb"][pi]
    dmap, mae = diff_map(gt, prgb)
    is_ctx = k <= 0
    tag = "context (decoded)" if is_ctx else "future"
    panels = [
        [label(resize(gt, pw, ph), "GT RGB"),
         label(resize(prgb, pw, ph), f"SUV RGB [{tag}]", color=CTX_COLOR if is_ctx else FUT_COLOR),
         label(label(resize(dmap, pw, ph), "|GT - SUV|"), f"MAE {mae:.1f}", corner="tr")],
        [label(resize(arrays["pred_depth"][pi], pw, ph), "SUV depth"),
         label(resize(arrays["pred_seg"][pi], pw, ph), "SUV seg"),
         label(resize(arrays["pred_instance"][pi], pw, ph), "SUV instance")],
    ]
    grid = np.concatenate([np.concatenate(r, axis=1) for r in panels], axis=0)
    body = np.concatenate([grid, bev], axis=1)

    header_h = 44
    W = body.shape[1]
    head = Image.new("RGB", (W, header_h), BG)
    d = ImageDraw.Draw(head)
    tstr = "t0" if k == 0 else f"t{k * DT:+.1f}s"
    phase = "CONTEXT" if is_ctx else "FUTURE"
    col = CTX_COLOR if is_ctx else FUT_COLOR
    d.rectangle([0, 0, 200, header_h], fill=col)
    d.text((12, 8), f"{tstr}", font=font(24), fill=(0, 0, 0))
    d.text((215, 4), f"{phase}", font=font(16), fill=col)
    d.text((215, 24), f"{meta['token']}  |  {meta.get('log_name', '')}", font=font(13), fill=(180, 180, 180))
    d.text((W - 520, 12), f"cmd: {command_name(meta['driving_command'])}   ADE {meta['ade_m']:.2f} m",
           font=font(18), fill=(255, 210, 127))
    # progress bar over all steps
    idx = frame_indices(meta)
    n = len(idx)
    bar_y = header_h - 4
    pos = [i for i, (kk, _, _) in enumerate(idx) if kk == k][0]
    for i, (kk, _, _) in enumerate(idx):
        x0 = int(i * W / n)
        x1 = int((i + 1) * W / n) - 2
        c = (CTX_COLOR if kk <= 0 else FUT_COLOR) if i <= pos else (60, 60, 60)
        d.rectangle([x0, bar_y, x1, header_h], fill=c)
    return np.concatenate([np.asarray(head), body], axis=0)


def render_mp4(arrays, meta, path: Path, scale: float, sec_per_step: float, fps: int, hold_end: float) -> None:
    H, W = arrays["gt_rgb"].shape[1:3]
    pw, ph = _even(int(W * scale)), _even(int(H * scale))
    frames = []
    idx = frame_indices(meta)
    for k, gi, pi in idx:
        bev = bev_panel(meta, _even(int(pw * 0.9)), 2 * ph, max(k, 0))
        frames.append(compose_frame(arrays, meta, k, gi, pi, pw, ph, bev))
    reps = max(1, int(round(sec_per_step * fps)))
    seq = []
    for i, f in enumerate(frames):
        r = reps + (int(round(hold_end * fps)) if i == len(frames) - 1 else 0)
        seq.extend([f] * r)
    write_mp4(path, seq, fps)


def render_grid(arrays, meta, path: Path, scale: float) -> None:
    H, W = arrays["gt_rgb"].shape[1:3]
    pw, ph = int(W * scale), int(H * scale)
    rows = [("GT RGB", "gt_rgb"), ("SUV RGB", "pred_rgb"), ("SUV depth", "pred_depth"),
            ("SUV seg", "pred_seg"), ("SUV instance", "pred_instance")]
    fut = [(k, gi, pi) for k, gi, pi in frame_indices(meta) if k > 0]
    lw, top = 150, 78
    canvas = Image.new("RGB", (lw + pw * len(fut), top + ph * len(rows)), BG)
    d = ImageDraw.Draw(canvas)
    d.text((10, 8), f"{meta['token']}   cmd: {command_name(meta['driving_command'])}   ADE {meta['ade_m']:.2f} m"
                    f"   ({meta.get('log_name', '')})", font=font(22), fill=FG)
    for c, (k, _, _) in enumerate(fut):
        d.text((lw + c * pw + pw // 2 - 30, top - 30), f"t+{k * DT:.1f}s", font=font(20), fill=FUT_COLOR)
    for r, (name, key) in enumerate(rows):
        d.text((10, top + r * ph + ph // 2 - 10), name, font=font(17), fill=FG)
        for c, (k, gi, pi) in enumerate(fut):
            img = arrays[key][gi if key == "gt_rgb" else pi]
            canvas.paste(Image.fromarray(resize(img, pw, ph)), (lw + c * pw, top + r * ph))
    for c in range(1, len(fut)):
        d.line([(lw + c * pw, top), (lw + c * pw, canvas.height)], fill=BG, width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def render_thumb(arrays, meta, path: Path, w: int = 320) -> None:
    sel = [(k, gi, pi) for k, gi, pi in frame_indices(meta) if k == 4] or [frame_indices(meta)[-1]]
    k, gi, pi = sel[0]
    H, W = arrays["gt_rgb"].shape[1:3]
    h = int(H * w / W)
    a = label(resize(arrays["gt_rgb"][gi], w, h), "GT", size=12)
    b = label(resize(arrays["pred_rgb"][pi], w, h), f"SUV t+{k * DT:.1f}s", size=12, color=FUT_COLOR)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate([a, b], axis=0)).save(path, quality=88)


def write_mp4(path: Path, frames: Sequence[np.ndarray], fps: int, crf: int = 20) -> None:
    h, w = frames[0].shape[:2]
    assert h % 2 == 0 and w % 2 == 0, (h, w)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "2", str(tmp)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for f in frames:
            proc.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed ({rc}) for {path}")
    os.replace(tmp, path)


def _even(x: int) -> int:
    return x - (x % 2)


def process_scene(args_tuple) -> dict:
    scene_dir, out, opts = args_tuple
    arrays, meta = layout.load_scene_dump(scene_dir)
    token = meta["token"]
    align = check_alignment(arrays, meta)
    render_mp4(arrays, meta, out / "mp4" / f"{token}.mp4", opts["mp4_scale"], opts["sec_per_step"],
               opts["fps"], opts["hold_end"])
    render_grid(arrays, meta, out / "grid" / f"{token}.png", opts["grid_scale"])
    render_thumb(arrays, meta, out / "thumbs" / f"{token}.jpg")
    return {"token": token, "alignment": align}


# --------------------------------------------------------------------------- selection + index
def load_all_meta(raw_root: Path) -> List[dict]:
    out = []
    for d in layout.list_scene_dumps(raw_root):
        with open(d / "meta.json") as f:
            m = json.load(f)
        out.append({"token": m["token"], "ade_m": float(m["ade_m"]), "log_name": m.get("log_name"),
                    "command": command_name(m["driving_command"])})
    return out


def select_scenes(metas: List[dict], n_each: int, seed: int) -> List[dict]:
    srt = sorted(metas, key=lambda m: (m["ade_m"], m["token"]))
    best = srt[:n_each]
    worst = srt[-n_each:][::-1]
    taken = {m["token"] for m in best + worst}
    rest = [m for m in srt if m["token"] not in taken]
    rng = np.random.default_rng(seed)
    rand_idx = sorted(rng.choice(len(rest), size=min(n_each, len(rest)), replace=False).tolist())
    rand = sorted([rest[i] for i in rand_idx], key=lambda m: m["ade_m"])
    out = []
    for group, ms in (("best_ade", best), ("worst_ade", worst), ("random", rand)):
        for m in ms:
            out.append({**m, "group": group})
    return out


def write_index(out: Path, selection: List[dict], criteria: dict, align_summary: str,
                name: str = "index.html") -> None:
    groups = [("best_ade", "Lowest ADE (best)"), ("worst_ade", "Highest ADE (worst)"),
              ("random", f"Random (seed {criteria.get('seed', 0)})"), ("custom", "Custom (--tokens)")]
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>SUV viz</title><style>",
        "body{background:#121214;color:#ddd;font-family:sans-serif;margin:20px}",
        "h1{font-size:20px}h2{font-size:16px;margin-top:28px;color:#ffd27f}",
        ".g{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}",
        ".c{background:#1d1d21;padding:6px;border-radius:6px}.c img{width:100%;display:block}",
        ".m{font-size:13px;margin-top:4px}.m a{color:#7fb3ff;margin-right:8px}code{color:#aaa}",
        "</style></head><body>",
        f"<h1>SUV rollouts vs GT — {html.escape(criteria.get('run', ''))}</h1>",
        f"<p>{html.escape(criteria.get('rule', ''))}</p>",
        f"<p>Frame alignment: {html.escape(align_summary)}</p>",
        "<p>mp4: GT RGB | SUV RGB | |GT−SUV| ; SUV depth | seg | instance ; BEV (pred vs GT). "
        "Grid: rows GT RGB, SUV RGB, depth, seg, instance × cols t+0.5…t+4.0 s.</p>",
    ]
    for g, title in groups:
        ms = [m for m in selection if m["group"] == g]
        if not ms:
            continue
        parts.append(f"<h2>{html.escape(title)} — {len(ms)}</h2><div class='g'>")
        for m in ms:
            t = html.escape(m["token"])
            parts.append(
                f"<div class='c'><a href='mp4/{t}.mp4'><img loading='lazy' src='thumbs/{t}.jpg'></a>"
                f"<div class='m'><b>ADE {m['ade_m']:.2f} m</b> · {html.escape(m['command'])}<br>"
                f"<code>{t}</code><br><a href='mp4/{t}.mp4'>mp4</a><a href='grid/{t}.png'>grid</a></div></div>")
        parts.append("</div>")
    parts.append("</body></html>")
    (out / name).write_text("\n".join(parts))


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokens", nargs="*", default=None, help="explicit scene tokens (skips default selection)")
    p.add_argument("--n-each", type=int, default=8, help="scenes per group: best / worst / random")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sec-per-step", type=float, default=1.0, help="seconds each 2 Hz step is shown (0.5 = real time)")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--hold-end", type=float, default=1.0, help="extra seconds on the last frame")
    p.add_argument("--mp4-scale", type=float, default=0.75)
    p.add_argument("--grid-scale", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args(argv)

    raw_root, out = Path(a.raw_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    metas = load_all_meta(raw_root)
    by_tok = {m["token"]: m for m in metas}
    run_name = raw_root.parent.name
    if a.tokens:
        missing = [t for t in a.tokens if t not in by_tok]
        if missing:
            raise SystemExit(f"tokens without a dump in {raw_root}: {missing}")
        selection = [{**by_tok[t], "group": "custom"} for t in a.tokens]
        criteria = {"run": run_name, "rule": f"custom token list ({len(selection)})", "seed": a.seed}
    else:
        selection = select_scenes(metas, a.n_each, a.seed)
        criteria = {
            "run": run_name,
            "rule": (f"From {len(metas)} rollouts in {raw_root}: {a.n_each} lowest ADE, {a.n_each} highest ADE "
                     f"(ADE = mean L2 over 8 future waypoints, meta.json ade_m), and {a.n_each} uniformly random "
                     f"from the remaining scenes with numpy default_rng(seed={a.seed})."),
            "seed": a.seed,
            "n_pool": len(metas),
            "ade_pool_mean": float(np.mean([m["ade_m"] for m in metas])),
            "ade_pool_median": float(np.median([m["ade_m"] for m in metas])),
        }

    opts = dict(mp4_scale=a.mp4_scale, grid_scale=a.grid_scale, sec_per_step=a.sec_per_step, fps=a.fps,
                hold_end=a.hold_end)
    jobs = [(raw_root / m["token"], out, opts) for m in selection]
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        results = list(ex.map(process_scene, jobs))
    align = {r["token"]: r["alignment"] for r in results}
    n_ok = sum(v["ok"] for v in align.values())
    align_summary = (f"pred[k+{1}] <-> gt[k] for history (pred[0] = repeat_first pad), last condition frame "
                     f"pred[4] = gt[3] = t0; future pred[5+i] <-> gt[4+i] = t+{DT}(i+1)s. "
                     f"Decoded-context argmin check passed {n_ok}/{len(align)} scenes.")
    for m in selection:
        m["alignment_ok"] = align[m["token"]]["ok"]
        m["context_mae"] = [round(x, 2) for x in align[m["token"]]["context_mae"]]
    sel_path = out / "selection.json"
    if a.tokens:
        sel_path = out / "selection_custom.json"
    sel_path.write_text(json.dumps({"criteria": criteria, "alignment": align_summary, "scenes": selection},
                                   indent=2))
    # --tokens runs never clobber the default 24-scene selection/index
    write_index(out, selection, criteria, align_summary, "index_custom.html" if a.tokens else "index.html")
    print(f"wrote {len(results)} scenes -> {out}  ({align_summary})")


if __name__ == "__main__":
    main()
