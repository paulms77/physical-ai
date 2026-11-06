# Phase 0 — 파이프라인 관통 (Pipeline Bring-up)

> 목표: **모델보다 배관을 먼저 완성한다.** iPhone(또는 mock) → WebSocket 수신 →
> 프레임 디코드 → ARKit→ROS 좌표 변환 → rerun 3D 시각화까지 데이터가 끝까지
> 흐르게 만든다. 완료 조건은 "폰을 들고 걸었을 때 화면 속 카메라가 따라 움직이는 것".

이 문서는 Phase 0에서 만든 코드의 **구조·실행법·학습 포인트**를 정리한다.

---

## 1. 데이터 흐름

```
 iPhone / mock_sender                    노트북 (수신·처리)
 ────────────────────                    ──────────────────────────────
 가상/실제 카메라 포즈                     FrameReceiver (WebSocket 서버)
   │ ARKit 규약 (Y-up, -Z 전방)                 │
   ▼                                            ▼
 encode_frame  ──[WebSocket 바이너리]──▶  decode_frame ─▶ Frame
 (JSON헤더+JPEG)                                │
                                                ▼
                                    arkit_camera_pose_to_ros
                                     (ARKit → ROS map 규약)
                                                │
                                                ▼
                                        RerunVisualizer
                                    (카메라 좌표축 + 궤적 + RGB)
```

## 2. 폴더 구조

```
pysical/
├── src/pysical/
│   ├── io/
│   │   ├── types.py        # Frame, CameraIntrinsics — 모듈 간 공용 계약
│   │   ├── protocol.py     # 와이어 포맷 encode/decode (iOS 앱이 따를 규약)
│   │   ├── receiver.py     # WebSocket 수신 서버 (콜백 방식)
│   │   └── mock_sender.py  # iPhone 대역 테스트 송신기
│   ├── geometry/
│   │   ├── transforms.py   # ★ ARKit↔ROS 축 변환 (가장 실수하기 쉬운 곳)
│   │   └── mock_camera.py  # look-at / orbit 가상 포즈 생성
│   ├── viz/
│   │   └── rerun_viz.py    # rerun 3D 시각화
│   └── utils/
│       ├── config.py       # config/*.yaml 로딩
│       └── logging.py      # 공통 로거
├── config/phase0.yaml      # 포트·fps·궤적 길이 등 (하드코딩 배제)
├── scripts/
│   ├── run_receiver.py     # 수신기 실행 (Phase 0 본체)
│   └── run_mock_sender.py  # mock 송신기 실행
├── tests/                  # 좌표변환·프로토콜 단위 테스트
└── data/recordings/        # .rrd 등 중간 산출물 (git 제외)
```

## 3. 설치

```bash
# Python 3.12 venv (uv가 사내망에서 막혀 표준 venv 사용, 결과 동일)
/usr/local/bin/python3.12 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pip install -r requirements-phase0.txt
```

## 4. 실행

**터미널 1 — 수신기 (rerun 뷰어가 뜨는 노트북)**
```bash
.venv/bin/python scripts/run_receiver.py
```
GUI가 없는 서버라면 헤드리스로 파일에 저장한 뒤 로컬에서 연다:
```bash
.venv/bin/python scripts/run_receiver.py --no-viewer --save data/recordings/run.rrd
```

**터미널 2 — mock 송신기 (iPhone이 아직 없을 때)**
```bash
.venv/bin/python scripts/run_mock_sender.py           # 원 궤적 카메라 20초
.venv/bin/python scripts/run_mock_sender.py --duration 0   # 무한
```

rerun 뷰어에서 카메라(좌표축 X빨강/Y초록/Z파랑)가 원을 그리며 돌고, 노란 궤적선이
그려지면 배관이 통한 것이다.

## 5. 테스트

```bash
.venv/bin/python tests/test_transforms.py   # ARKit→ROS 축 변환 (5개)
.venv/bin/python tests/test_protocol.py      # 프로토콜 왕복 + mock 카메라 (4개)
```

## 6. 학습 포인트 (직무지원 관점)

- **좌표계 변환이 핵심이다.** `transforms.py`가 이 프로젝트의 심장이다. ARKit(Y-up,
  카메라 -Z 전방)과 ROS(REP-103: map은 X-앞/Y-왼쪽/Z-위, optical은 X-오른쪽/Y-아래/Z-앞)의
  규약 차이를 두 단계 변환(월드 축 + 카메라 축)으로 처리한다. 부호 실수가 치명적이라
  '알려진 방향'을 넣는 테스트로 못박았다.
- **관심사 분리.** 수신(네트워크) / 변환(수학) / 시각화(렌더)가 각각의 모듈이고,
  `Frame` 하나로만 연결된다. Phase 1에서 검출기를 붙일 때 이 사이에 끼워 넣으면 된다.
- **iOS 앱 없이 검증 가능한 구조.** `mock_sender`가 실제 앱과 **똑같은 와이어 프로토콜**을
  쓰므로, 앱을 만들기 전에 노트북 파이프라인을 전부 검증할 수 있다. 나중에 iOS 앱은
  이 프로토콜만 지키면 그대로 붙는다.
- **환경 이식성.** 이 코드는 CPU만 쓰므로 A6000 서버에서 개발하지만, 그대로 추론
  노트북으로 옮겨 돌아간다(README의 배포 구조와 일치).

## 7. 다음 단계 (Phase 1 예고)

- 검출 모델(YOLOE/YOLO-World, prompt-then-detect)을 `Frame.rgb`에 붙인다.
- 마스크에서 접지점 픽셀 (u,v)을 뽑고, intrinsics로 카메라 광선을 만든다.
- 방법 A(평면 교차)로 첫 world 좌표를 만들고, 같은 픽셀의 sceneDepth와 비교한다.
- iOS 앱: ARKit 포즈 + `SFSpeechRecognizer` 음성 프롬프트를 이 프로토콜로 전송.
