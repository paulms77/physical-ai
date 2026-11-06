"""프로젝트 전역에서 주고받는 데이터 타입 정의.

Phase 0의 데이터 흐름은 아주 단순하다::

    iPhone ──(WebSocket)──▶ Frame ──▶ 좌표변환 ──▶ rerun 시각화

이 모듈은 그 흐름을 타고 다니는 **한 프레임(Frame)** 의 형태를 한곳에 못박아,
수신기·좌표변환·시각화가 같은 계약(contract)을 공유하도록 한다. 필드가 바뀌면
여기만 고치면 되고, 나머지 모듈은 이 타입만 바라본다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ARKit이 카메라 포즈를 보낼 때 쓰는 tracking state. 이 문자열은 iPhone 앱(ARKit)
# 쪽의 `ARCamera.trackingState`를 그대로 옮긴 값이라고 가정한다.
TRACKING_NORMAL = "normal"
TRACKING_LIMITED = "limited"
TRACKING_NOT_AVAILABLE = "notAvailable"


@dataclass
class CameraIntrinsics:
    """카메라 내부 파라미터 (핀홀 모델).

    픽셀 (u, v)를 카메라 좌표계의 광선으로 되돌릴 때 쓴다. Phase 1의 depth→3D
    변환에서 본격적으로 쓰이고, Phase 0에서는 프레임에 담겨 오기만 한다.
    """

    fx: float
    fy: float
    cx: float
    cy: float

    def as_matrix(self) -> np.ndarray:
        """3x3 K 행렬로 변환한다."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def scaled_to(self, src_size: tuple[int, int], dst_size: tuple[int, int]) -> "CameraIntrinsics":
        """다른 해상도로 intrinsics를 스케일링한다.

        ARKit LiDAR depth(256×192)와 RGB(예: 640×480)는 해상도가 다르다. depth를
        3D로 역투영하려면 depth 해상도에 맞는 K가 필요하므로, RGB 기준 K를 depth
        크기로 선형 스케일한다. 핀홀 모델에서 K의 각 성분은 픽셀 스케일에 비례한다.

        Parameters
        ----------
        src_size, dst_size:
            (width, height) 형식. 이 intrinsics가 기준으로 삼는 크기(src)와 목표 크기(dst).
        """
        sx = dst_size[0] / src_size[0]
        sy = dst_size[1] / src_size[1]
        return CameraIntrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
        )


@dataclass
class Frame:
    """iPhone이 한 시점에 보내는 센서 묶음.

    Attributes
    ----------
    timestamp:
        iPhone 기준 촬영 시각(초). Mac/노트북 시계와 다를 수 있어 동기화 이슈의
        원인이 된다(README §11 참고). Phase 0에서는 순서 정렬용으로만 쓴다.
    T_world_camera:
        ARKit world 좌표계 기준 카메라의 4x4 포즈 행렬. **ARKit 규약**(Y-up,
        카메라가 보는 방향 -Z)을 그대로 담는다. ROS 규약으로 바꾸는 건 geometry
        모듈의 몫이며, 이 타입은 원본을 손대지 않는다.
    rgb:
        디코드된 RGB 이미지 (H, W, 3), uint8. 포즈만 볼 때는 None일 수 있다.
    intrinsics:
        카메라 내부 파라미터. RGB가 없으면 None일 수 있다.
    tracking_state:
        ARKit tracking 상태 문자열. ``limited``면 포즈 신뢰도가 떨어진다.
    depth_lidar:
        ARKit sceneDepth (LiDAR) depth map (H_d, W_d) float32, **미터 단위**.
        RGB와 해상도가 다를 수 있다(보통 256×192). 없으면 None.
        → 관측기반 표면추정(방법 A′)과 정답지에 쓴다(docs/ARCHITECTURE_v2.md §1-D).
    depth_confidence:
        depth_lidar와 같은 해상도의 신뢰도 맵 (H_d, W_d) uint8. ARKit은 0~2(또는
        문서상 1~3) 값을 준다. 낮은 신뢰도 픽셀을 평면 fitting에서 거를 때 쓴다.
    depth_mono:
        노트북에서 monocular depth 모델로 추정한 depth (H, W) float32, 미터.
        iPhone이 보내는 게 아니라 수신 후 파이프라인이 채운다(방법 B). 없으면 None.
    depth_intrinsics:
        depth_lidar 해상도에 맞춘 카메라 intrinsics. depth 역투영에 쓴다. LiDAR가
        RGB와 다른 해상도이므로 :meth:`CameraIntrinsics.scaled_to` 로 만들어 담는다.

    Notes
    -----
    depth 관련 필드는 전부 Optional(기본 None)이라, Phase 0 코드와 완전히 하위
    호환된다. LiDAR가 없는 기기·mock에서도 Frame은 그대로 성립한다.
    """

    timestamp: float
    T_world_camera: np.ndarray  # (4, 4) float64, ARKit 규약
    rgb: np.ndarray | None = None
    intrinsics: CameraIntrinsics | None = None
    tracking_state: str = TRACKING_NORMAL
    # --- depth / scene geometry (Phase 1+에서 사용, docs/ARCHITECTURE_v2.md §4) ---
    depth_lidar: np.ndarray | None = None
    depth_confidence: np.ndarray | None = None
    depth_mono: np.ndarray | None = None
    depth_intrinsics: CameraIntrinsics | None = None

    def __post_init__(self) -> None:
        # 포즈는 반드시 4x4여야 이후 계산이 성립한다. 잘못된 데이터가 조용히
        # 흘러가면 시각화가 이상해지고 원인 추적이 어려우므로 여기서 막는다.
        arr = np.asarray(self.T_world_camera, dtype=np.float64)
        if arr.shape != (4, 4):
            raise ValueError(
                f"T_world_camera는 4x4여야 합니다. 받은 shape={arr.shape}"
            )
        self.T_world_camera = arr

    @property
    def camera_position_world(self) -> np.ndarray:
        """ARKit world 좌표계에서 카메라의 위치 (x, y, z)."""
        return self.T_world_camera[:3, 3]

    @property
    def is_tracking_ok(self) -> bool:
        """포즈를 믿어도 되는 상태인지."""
        return self.tracking_state == TRACKING_NORMAL

    @property
    def has_lidar_depth(self) -> bool:
        """LiDAR sceneDepth가 실려 있는지 (방법 A′·정답지 사용 가능 여부)."""
        return self.depth_lidar is not None

    @property
    def has_any_depth(self) -> bool:
        """LiDAR든 mono든 depth가 하나라도 있는지."""
        return self.depth_lidar is not None or self.depth_mono is not None
