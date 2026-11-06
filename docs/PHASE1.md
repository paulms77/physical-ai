# Phase 1 — 인식 + 관측기반 3D 위치추정

> 목표: 검출된 물체의 3D world 좌표를, **미리 아는 평면 없이** 관측된 depth에서
> 표면을 추정해 만든다. docs/ARCHITECTURE_v2.md의 파이프라인을 코드로 구현한다.

이 문서는 Phase 1에서 만든 코드의 구조·상태·학습 포인트를 정리한다.

---

## 1. 현재 상태 (2026-09-16)

**완성(순수 기하, 학습·GPU 불필요) — 전부 테스트 통과:**
- `geometry/backproject.py` — depth + K → 카메라 좌표계 3D 점 (역투영)
- `geometry/plane_fit.py` — NumPy RANSAC 평면 fitting + 점 투영
- `perception/localize.py` — mask/bbox + frame → world 좌표 (세 depth 경로)
- `perception/detector.py` — 검출기 인터페이스 + 한↔영 프롬프트 매핑 + mock 검출기

**남은 것 (실제 모델 연결):**
- YOLOE(ONNX/CPU) 검출기를 `BaseDetector` 인터페이스로 구현
- (선택) MobileSAM으로 mask 정밀화
- Depth Anything V2(ONNX) 로 `frame.depth_mono` 채우기 → 방법 B 실측
- iOS 앱: ARKit `sceneDepth` + `smoothedSceneDepth` 전송, `SFSpeechRecognizer` 음성

## 2. 관측기반 위치추정 흐름

```
Detection(mask/bbox)  +  Frame(depth, pose, K)
        │
        ▼  ground_pixel()  — 접지점 = 물체 최하단 (중심 아님)
   접지점 픽셀 (u,v)
        │
   ┌────┴───────────────┬────────────────────┐
   ▼ METHOD_PLANE (A′)   ▼ METHOD_DEPTH (B)    ▼ METHOD_LIDAR (정답지)
 표면영역 depth→3D점      접지점 depth 직접      LiDAR depth 직접
 RANSAC 평면 fitting      →3D점                 →3D점
 접지점을 평면에 투영
        └────────────┬───────────────────────┘
                     ▼  camera→ARKit world→ROS map (transforms.py)
              ObjectLocation(position_world, method, inlier_ratio)
```

세 경로가 **같은 접지점·같은 K·같은 좌표변환**을 공유한다. 그래서 결과 차이가 순수하게
depth/표면추정 차이에서만 나온다(README §5 원칙). Phase 3 비교 실험의 기반.

## 3. 핵심 설계 결정

- **방법 A의 진화(A′).** 기존 "미리 아는 ARKit 평면과 ray 교차" → "관측 depth에서
  RANSAC로 fitting한 평면에 투영". 처음 보는 책상에도 성립한다.
- **RANSAC은 NumPy 직접 구현.** Open3D는 dash/flask/jupyter 등 60개 의존성을 끌고 와
  과하다. 평면 fitting은 20~30줄이면 되고, 학습 목적상 내부가 보이는 게 낫다.
- **접지점 = 최하단.** bbox 중심을 쓰면 물체 높이의 절반만큼 앞으로 치우친다.
  마스크가 있으면 마스크 최하단, 없으면 bbox 하단변 중앙.
- **검출기는 인터페이스 우선.** `BaseDetector` Protocol + `MockDetector`로 배관을 먼저
  검증하고, YOLOE는 나중에 갈아끼운다(Phase 0의 mock 송신기 전략과 동일).

## 4. 검증

합성 3D 장면(바닥 평면 + 그 위 물체)을 카메라로 렌더한 depth로, 파이프라인이 물체의
world 좌표를 되찾는지 확인한다(`tests/test_localize.py`):
- LiDAR 경로: 정답 대비 **< 15cm**
- 관측평면(A′) 경로: 평면 fitting(inlier > 0.5) 후 **< 20cm**

전체 테스트 30개 통과 (transforms 5 · protocol 8 · geometry 7 · localize 4 · detector 6).

```bash
for t in tests/test_*.py; do .venv/bin/python "$t"; done
```

## 4-1. 내부 검증 체계 — iOS 앱/실기기 전에 어떻게 확인하나

실기기(iPhone) 라이브 데모 **전에** 아래 4단계로 내려가며 검증한다. 각 단계가
그 아래 단계를 신뢰할 수 있게 만든다. 지금 ①~③이 갖춰졌다.

| 단계 | 입력 | 검증 대상 | 도구 | 상태 |
|---|---|---|---|---|
| ① 합성 장면 | 수학으로 만든 depth/포즈 | 기하·좌표변환 | tests/test_localize.py | ✅ |
| ② 정지 이미지 | 사진 1장(+depth) | **검출 모델**·접지점·3D | scripts/detect_image.py | ✅ |
| ③ 녹화 재생 | 동영상+포즈+depth(.pyrec) | 전체 파이프라인 시계열 | scripts/replay_recording.py | ✅ |
| ④ 실기기 라이브 | 실시간 WebSocket 스트림 | 데모 | scripts/run_receiver.py | iOS 앱 후 |

**핵심: "동영상만"으로는 안 된다.** 3D 위치추정은 각 프레임의 **카메라 포즈(ARKit)와
depth(LiDAR)** 가 영상과 함께 있어야 성립한다. 그래서 녹화 포맷(.pyrec)은 프레임마다
RGB·포즈·depth·intrinsics를 한 묶음으로 저장한다(io/recording.py). 이것이 iOS 앱의
첫 기능이 '녹화 버튼'인 이유다(README §8: "앱이 곧 수집 도구").

**③이 라이브와 같은 코드 경로인 이유.** 녹화는 WebSocket 프로토콜(encode_frame)을
그대로 재활용해 저장하고, 재생은 decode_frame으로 복원한다. 즉 재생에서 되는 것은
실기기 라이브에서도 된다 — 검증의 신뢰도가 여기서 온다.

```bash
# ② 사진 한 장으로 검출 + 3D (LiDAR 없이 mono depth)
.venv/bin/python scripts/detect_image.py --image desk.jpg --prompt "사과 컵" \
    --mono-depth --fx 900 --fy 900 --cx 405 --cy 540

# ③ 녹화 재생으로 전체 파이프라인 시계열 검증
.venv/bin/python scripts/replay_recording.py --rec desk.pyrec --prompt "사과 컵"
```

## 4-2. 실제 모델 상태 (2026-09-16)

- **검출: YOLOE-11s-seg** (ultralytics). prompt-then-detect 작동, CPU 0.21s/장,
  bbox+mask 반환. 실사진(bus.jpg)에서 person 4 + bus 1 정확히 검출 확인.
- **depth: Depth Anything V2 Small indoor (ONNX)**. onnxruntime CPU 0.14s/장.
  사내망 HF CDN 차단으로 GitHub(fabio-sim/Depth-Anything-ONNX)에서 확보.
  출력은 상대 depth(inverse) — metric은 LiDAR로 보정(calibrate_scale).
- 하이브리드 방침: 지금은 검증 위해 .pt/ONNX 직접, 최종 배포는 A6000 export → 노트북 onnxruntime.

## 5. 학습 포인트 (직무지원)

- **역투영·평면 fitting은 로봇 비전의 기본기다.** 핀홀 모델 역투영과 RANSAC은
  "카메라가 본 픽셀을 3D 세계로 되돌리는" 가장 기초적이고 중요한 연산이다.
  이걸 직접 구현해 봤다는 게 포트폴리오에서 의미 있다.
- **오차 전파를 잴 수 있는 구조.** 검출 bbox가 접지점을 정하고, 접지점이 3D 좌표를
  정한다. 그래서 "bbox가 몇 픽셀 틀리면 1m에서 몇 cm 틀리는지"를 이 파이프라인으로
  직접 측정할 수 있다(README §8, Phase 3).
- **CPU 친화 설계.** 무거운 부분(검출·depth)은 ONNX로, 기하 부분은 순수 NumPy로.
  표면추정에 학습을 안 쓰므로 CPU에서 사실상 공짜다.
