# Phase 1~3 재설계 — 관측기반 3D 표면 이해 (2026)

> 개정 2026-09-16. 기존 README의 방법 A("미리 검출된 ARKit 평면과 ray 교차")를
> **"표면을 관측해서 평면을 추정한다"** 로 재설계한다. 핵심 전환:
> *책상 좌표를 미리 안다* → *책상이라는 표면을 그 자리에서 관측·추정한다*.

---

## 0. 왜 바꾸는가

기존 README의 방법 A는 ARKit이 미리 찾아둔 `ARPlaneAnchor`(수평 평면)에 광선을
쏘아 교차점을 얻는 방식이었다. 이건 두 가지 한계가 있다.

1. **평면이 미리 잡혀 있어야 한다.** ARKit 평면 검출은 폰을 몇 초 흔들어야 수렴하고,
   처음 보는 배치에서는 지연이 생긴다(README §11).
2. **"그 물체가 놓인 표면"과 "ARKit이 찾은 평면"이 같다는 보장이 없다.** 여러 평면이
   있으면 엉뚱한 평면과 교차할 수 있다.

2026년 기준 최신 파이프라인은 대신 **관측된 depth에서 그 물체 주변의 표면을 직접
추정**한다. 데이터 흐름은 사용자가 제안한 형태 그대로다::

    RGB + Camera Pose
          ├── Object Detection / Segmentation → object mask
          ├── Depth / 3D Geometry            → per-pixel 3D points
          └── Surface Understanding          → 물체가 놓인 평면을 RANSAC로 fitting
                     ▼
            Object 3D Position → World Frame

이렇게 하면 책상이 처음 보는 위치에 있어도, segmentation으로 책상 영역을 찾고 →
그 영역의 depth 3D point를 모아 → RANSAC로 평면을 fitting하고 → 물체의 접지점을
그 평면에 투영하면 된다. **평면을 아는 게 아니라 관측한다.**

---

## 1. 2026 최신 방식 비교 (난이도·정확도·CPU 적합성)

우리 제약: **추론은 GPU 없는 노트북 CPU**, 검출은 2~5Hz + 트래커 보간(README §5).
따라서 "CPU에서 도는가"가 정확도만큼 중요한 축이다.

### 1-A. 세그멘테이션 (물체 마스크)

| 모델 | 정확도 | CPU 속도 | 난이도 | 판정 |
|---|---|---|---|---|
| **MobileSAM** | SAM 대비 mIoU ~2–4p↓ (≈0.74) | **CPU에서 부드러움**, SAM 대비 5–38× 빠름, 66× 작음 | 낮음 | ✅ **1순위**. CPU 프롬프트 세그멘테이션 |
| FastSAM | MobileSAM과 비슷 | GPU 40ms (MobileSAM 10ms의 4배 느림) | 낮음 | △ MobileSAM보다 느려 제외 |
| SAM 2 | 최고, 비디오 트래킹 내장 | M4에서 ~44fps지만 그건 Apple Silicon 얘기, x86 CPU엔 무거움 | 중 | ⏸ Phase 3 정확도 비교용 후보 |

**결정:** 실시간 경로는 **MobileSAM**. 단, 매 프레임 SAM을 돌리지 않는다 — 검출기
(YOLOE)가 bbox를 주고, 그 bbox를 프롬프트로 MobileSAM에 한 번만 넣어 마스크를 얻는다.
YOLOE가 세그멘테이션까지 내주면(YOLOE-seg) SAM을 아예 생략할 수도 있다.

### 1-B. Monocular Metric Depth (방법 B)

| 모델 | 정확도(zero-shot metric) | CPU 속도 | ONNX | 판정 |
|---|---|---|---|---|
| **Depth Anything V2 (small)** | ETH3D 36.3 / KITTI 7.4 | 셋 중 가장 가벼움 | ✅ | ✅ **CPU 1순위** (small/base) |
| UniDepthV2 | ETH3D 25.3 / KITTI 4.2, intrinsics까지 추정 | 무거움 | ✅ | ⏸ 정확도 비교용 |
| Metric3D v2 | **ETH3D 87.7 / KITTI 3.9 (1위)** | 가장 무거움 | 부분 | ⏸ 정확도 상한 참조 |

*(수치는 벤치마크마다 척도가 달라 절대비교가 아니라 상대 순위로 읽는다. Metric3Dv2가
전반적으로 가장 정확하고, Depth Anything V2가 정확도-속도 균형이 가장 좋다.)*

**결정:** 실시간 방법 B는 **Depth Anything V2 small (ONNX/CPU)**. 무거우면 입력을
256~384px로 낮추거나 depth만 저주기(5Hz)로 돌린다. Metric3Dv2/UniDepthV2는 Phase 3에서
"정확도 상한이 어디인가"를 A6000 오프라인 배치로 재는 참조군.

### 1-C. 표면 이해 (평면 추정)

| 방법 | 정확도 | CPU 속도 | 난이도 | 판정 |
|---|---|---|---|---|
| **Open3D `segment_plane` (RANSAC)** | depth 품질에 좌우, robust | **순수 기하, CPU 사실상 공짜** | 낮음 | ✅ **채택** |
| pyRANSAC-3D | 동등 | 동등 | 낮음 | ○ 대안(의존성 가벼움) |
| 학습기반 plane detection | 더 매끈 | 무거움 | 높음 | ✗ 과함 |

**결정:** **Open3D RANSAC `segment_plane`**. 물체 접지 영역 주변의 3D point를 모아
`ax+by+cz+d=0` 평면을 fitting한다. 학습이 필요 없어 CPU에서 무료에 가깝다.

### 1-D. Depth 소스 — LiDAR vs mono (둘 다 지원)

| 소스 | 해상도 | 정확도 | 비고 |
|---|---|---|---|
| **ARKit sceneDepth (LiDAR)** | 256×192 + confidence(1~3) | 높음(근거리) | `smoothedSceneDepth`로 노이즈↓. iPhone→노트북 전송 |
| **monocular depth** | RGB 해상도 | 스케일 드리프트 | LiDAR로 스케일 보정 가능 |

**결정(사용자 방침):** Frame이 **둘 다 담고**, 표면추정은 가용한 소스를 쓰되 **둘을
비교**한다. LiDAR가 있으면 주력 + 정답지 겸용, 없으면 mono로 폴백. 이 구조가 README의
"LiDAR 없이 얼마나 되는가"라는 질문을 그대로 살린다.

---

## 2. 관측기반 표면추정 파이프라인 (핵심)

물체 하나의 3D 위치를 얻는 절차. 세 depth 경로가 **똑같은 픽셀에서 출발**하는
README §5의 원칙을 유지한다.

```
1. Detect     YOLOE(prompt-then-detect)로 bbox + (가능하면) mask
2. Segment    필요시 MobileSAM으로 정밀 mask (bbox 프롬프트)
3. Backproject  mask 영역 + 그 아래 표면 영역의 픽셀을 depth로 3D 점 구름으로 역투영
                (intrinsics K 사용, camera 좌표계)
4. Fit surface  물체 하단 주변 3D point에 Open3D RANSAC → 지지 평면 π 추정
5. Ground point 물체 mask의 최하단(접지점) 픽셀을 3D로 → 평면 π에 투영
                (평면 위 없는 물체는 depth 3D점 자체를 사용 — 방법 B가 이때 빛난다)
6. To world     camera 좌표 → ARKit world → ROS map (기존 transforms.py 재사용)
```

**중요한 설계점 — 방법 A의 진화.**
- 기존 A: 미리 아는 평면과 ray 교차 → 폐기
- 새 A′: **관측 depth에서 fitting한 평면**에 접지점 투영 → 평면이 처음 보여도 성립
- 방법 B: 평면 자체를 안 쓰고 물체 픽셀의 depth 3D점을 직접 사용 → 공중/선반 물체 커버
- 정답지: LiDAR depth로 같은 절차 → A′, B를 cm 단위로 채점

세 경로가 같은 mask·같은 픽셀·같은 `transforms.py`를 공유하므로, 오차가 순수하게
depth/표면추정 차이에서만 나온다.

---

## 3. 재설계된 Phase 1~3

### Phase 1 — 인식 + 관측기반 첫 좌표
- YOLOE(ONNX/CPU, prompt-then-detect) 붙이기. 음성/텍스트 프롬프트 → 한↔영 매핑.
- (선택) MobileSAM으로 mask 정밀화.
- Frame에 실려온 **LiDAR depth로** backproject → RANSAC 평면 fitting → 접지점 투영.
- 같은 픽셀에서 mono depth도 계산해 **A′ vs B vs LiDAR** 3자 비교 시작.
- **완료 조건:** 처음 보는 책상에 물체를 놓아도, 미리 평면을 지정하지 않고 3D 좌표가 나온다.

### Phase 2 — ROS2 통합 (기존 계획 유지)
- ROS2 Foxy 네이티브, TF 트리(map→base_link→camera→object), `/object_target` PoseStamped.
- 추정된 평면 π를 RViz `Marker`(면)로도 시각화 → "무엇을 표면으로 봤는지" 눈으로 확인.

### Phase 3 — 비교 실험 + A6000 학습
- 인식 축: fine-tuned YOLOv11n(closed) vs YOLOE(open). A6000에서 학습·자동라벨링·ONNX.
- 위치 축: **A′(관측평면 투영) vs B(mono depth 직접) vs LiDAR 정답지**.
  - 추가 지표: 평면 fitting 품질(inlier 비율, 평면까지 잔차), viewpoint-invariance.
- depth 상한 참조: Metric3Dv2/UniDepthV2를 A6000 오프라인 배치로 돌려 "mono의 천장" 표시.

---

## 4. 이 재설계가 Frame 구조에 요구하는 것

기존 `Frame`(timestamp, T_world_camera, rgb, intrinsics, tracking_state)에 **depth와
scene geometry 자리**를 추가해야 위 파이프라인이 성립한다.

- `depth_lidar`: (H_d, W_d) float32 미터, ARKit sceneDepth. None 가능.
- `depth_confidence`: (H_d, W_d) uint8, ARKit confidence(0~2 또는 1~3). None 가능.
- `depth_mono`: (H, W) float32, 노트북에서 추정. Frame 생성 후 채워질 수도 있음.
- intrinsics는 이미 있음 — depth 해상도와 RGB 해상도가 달라(256×192 vs 640×480)
  **스케일 보정**이 필요하니, depth용 intrinsics 스케일링 헬퍼를 geometry에 둔다.

→ 이 확장은 다음 커밋에서 `io/types.py`에 반영한다(하위 호환: 전부 Optional, 기본 None).

---

## 5. 요약 결정표

| 역할 | 실시간(노트북 CPU) | 정확도 참조(A6000 오프라인) |
|---|---|---|
| 검출 | YOLOE (prompt-then-detect, ONNX) | fine-tuned YOLOv11n |
| 세그멘테이션 | MobileSAM 또는 YOLOE-seg | SAM 2 |
| depth | ARKit LiDAR ‖ Depth Anything V2 small | Metric3D v2 / UniDepthV2 |
| 표면추정 | Open3D RANSAC segment_plane | — (기하, 학습 없음) |
| 좌표변환 | transforms.py (ARKit→ROS) | 동일 |

---

## 참고 (Sources)

- MobileSAM / FastSAM / SAM2 CPU 속도: Ultralytics Docs (MobileSAM, SAM-2),
  "Faster Segment Anything" (arXiv:2306.14289), Efficient SAM Variants Survey (arXiv:2410.04960)
- Monocular metric depth: Depth Anything V2, UniDepthV2 (arXiv:2502.20110),
  Metric3D v2 (arXiv:2404.15506), Survey on Monocular Metric Depth (arXiv:2501.11841)
- 평면 fitting: Open3D `segment_plane` (RANSAC), pyRANSAC-3D (github: leomariga/pyRANSAC-3D)
- ARKit depth: `sceneDepth`/`smoothedSceneDepth` + confidence(1~3), 256×192 (Apple ARKit, 관련 오픈소스)
