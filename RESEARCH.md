# SUV 연구

## 목표
- 공개 SUV 체크포인트(suv_navsim.pt + Wan2.2-TI2V-5B)를 GPU 0 한 장에서 NAVSIM v1/v2(navtest, navhard) 로 재현 평가한다.
- SUV 가 생성하는 미래 영상(RGB·depth·seg·track)을 WorldLens 벤치마크(https://github.com/worldbench/worldlens)로 평가한다.

## 환경·경로
- GPU: 0번만 사용 (CUDA_VISIBLE_DEVICES=0). RTX 4090 24GB.
- 데이터·체크포인트 루트: /data2/woosung/datasets/SUV
  - 2026-11-01 기준 이 컨테이너에서 /data2 는 ro 마운트 → 사람이 rw 로 바꿔야 한다 (요청함).
- 시스템 nvcc 12.4, Ubuntu 22.04. WorldLens 는 py3.9 + torch2.1/cu118 + mmcv1.6(컴파일) 기준 → conda 안 cuda-nvcc 11.8 로 해결 시도, 안 되면 새 컨테이너 필요.

## 가설·질문
- H1: 단일 4090 에서 README 수치(v1 PDMS 90.8, v2 navtest EPDMS 91.0, navhard 36.9)를 ±0.3 안으로 재현한다.
- Q1: WorldLens 는 nuScenes 6-view 기준. SUV 는 NAVSIM 전방 1캠 → 어떤 차원(Generation·Reconstruction·Action-Following·Downstream)을
  그대로/수정해서 쓸 수 있는지 정해야 한다 (wl-integrate 의 조사 결과로 결정).

## 백로그
1. [진행] setup: env·체크포인트·NAVSIM 데이터·metric cache, 1-scene smoke  (tasks/setup.md)
2. [진행] wl-integrate: WorldLens 설치 + SUV 출력 → WorldLens 입력 어댑터, 적용 가능 차원 정리  (tasks/wl-integrate.md)
3. [대기] navsim-eval: v1 PDMS / v2 navtest / navhard 전체 평가 (setup 이후, GPU-h 추정 후 승인)
4. [대기] wl-eval: WorldLens 지표 실행 (2 이후)

## 결정
- 2026-11-01: 데이터 위치 /data2/woosung/datasets/SUV (사용자 지시). GPU 0 전용.

## GPU 예산
- 미정. 전체 평가 run 은 setup 의 smoke 로 scene 당 시간을 잰 뒤 GPU-h 를 추정해 승인받는다.
