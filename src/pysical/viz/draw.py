"""검출 결과를 이미지 위에 그리는 유틸 (내부 검증·디버깅용).

②단계 검증(정지 이미지)에서 "모델이 무엇을 어디서 찾았는지"를 눈으로 확인하려면
bbox·mask·라벨을 이미지에 얹어야 한다. rerun(3D)과 별개로, 2D 오버레이는 이 함수가 맡는다.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..perception.types import Detection, ObjectLocation


def draw_detections(
    rgb: np.ndarray,
    detections: list[Detection],
    locations: dict[int, dict[str, ObjectLocation]] | None = None,
) -> np.ndarray:
    """RGB 이미지에 검출 bbox/mask/라벨(+3D 위치)을 그려 새 이미지를 돌려준다.

    Parameters
    ----------
    rgb:
        (H, W, 3) uint8 원본 이미지. 원본은 손대지 않고 복사본에 그린다.
    detections:
        그릴 검출 목록.
    locations:
        {검출 인덱스: {method: ObjectLocation}}. 있으면 라벨에 3D 좌표를 붙인다.
    """
    canvas = rgb.copy()
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox_xyxy.astype(int)
        color = (0, 200, 0)

        # 마스크를 반투명으로 얹는다.
        if det.mask is not None:
            overlay = canvas.copy()
            overlay[det.mask] = color
            canvas = cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # 접지점(위치추정의 기준)을 표시 — 물체 최하단.
        gu, gv = det.ground_pixel().astype(int)
        cv2.circle(canvas, (int(gu), int(gv)), 4, (0, 0, 255), -1)

        label = f"{det.label} {det.score:.2f}"
        if locations and i in locations:
            # 대표로 하나의 좌표를 라벨에 붙인다(있으면 plane, 없으면 아무거나).
            locs = locations[i]
            key = "plane" if "plane" in locs else next(iter(locs))
            p = locs[key].position_world
            label += f"  [{key}] ({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})m"

        _put_label(canvas, label, x1, y1)
    return canvas


def _put_label(img: np.ndarray, text: str, x: int, y: int) -> None:
    """가독성 있게 배경 박스를 깔고 텍스트를 그린다."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.5, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    y = max(y, th + 4)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y), (0, 200, 0), -1)
    cv2.putText(img, text, (x + 2, y - 2), font, scale, (0, 0, 0), thick, cv2.LINE_AA)
