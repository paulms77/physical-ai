"""인식/위치추정 파이프라인의 데이터 타입.

검출기(YOLOE 등)가 내는 결과(Detection)와, 위치추정이 내는 결과(ObjectLocation)를
한곳에 정의한다. 검출기 구현이 무엇이든(YOLOE/YOLOv11n/mock) 이 타입만 내면 뒤
단계가 그대로 동작하도록 하는 계약이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# 위치추정에 쓴 depth 경로. docs/ARCHITECTURE_v2.md의 세 경로.
METHOD_PLANE = "plane"        # A′: 관측 평면에 접지점 투영
METHOD_DEPTH = "depth"        # B : 물체 픽셀 depth 직접 사용
METHOD_LIDAR = "lidar"        # 정답지: LiDAR depth 직접 사용


@dataclass
class Detection:
    """한 물체의 2D 검출 결과.

    Attributes
    ----------
    label:
        클래스 이름(예: "apple"). 프롬프트로 지정한 대상.
    score:
        검출 신뢰도 0~1.
    bbox_xyxy:
        (x1, y1, x2, y2) 픽셀 바운딩 박스.
    mask:
        (H, W) bool 세그멘테이션 마스크. 없으면 None(그땐 bbox로 대체).
    """

    label: str
    score: float
    bbox_xyxy: np.ndarray
    mask: np.ndarray | None = None

    def ground_pixel(self) -> np.ndarray:
        """접지점 픽셀 (u, v)을 추정한다.

        물체가 표면에 닿는 지점은 **물체의 최하단**이다. bbox 중심을 쓰면 물체
        높이의 절반만큼 앞으로 치우친다(README §5). 마스크가 있으면 마스크 최하단
        행의 가로 중앙을, 없으면 bbox 하단변의 중앙을 접지점으로 쓴다.
        """
        if self.mask is not None and self.mask.any():
            ys, xs = np.nonzero(self.mask)
            bottom_y = ys.max()
            bottom_xs = xs[ys == bottom_y]
            return np.array([bottom_xs.mean(), float(bottom_y)])
        x1, y1, x2, y2 = self.bbox_xyxy
        return np.array([(x1 + x2) / 2.0, float(y2)])


@dataclass
class ObjectLocation:
    """한 물체의 3D 위치추정 결과.

    Attributes
    ----------
    label, score:
        원본 검출 정보.
    position_world:
        (3,) ROS map 좌표계 3D 위치(미터). 파이프라인의 최종 산출물.
    method:
        위치를 만든 depth 경로(METHOD_PLANE / DEPTH / LIDAR).
    plane_inlier_ratio:
        방법 A′일 때 사용한 평면의 inlier 비율(품질 지표). 아니면 None.
    """

    label: str
    score: float
    position_world: np.ndarray
    method: str
    plane_inlier_ratio: float | None = None
