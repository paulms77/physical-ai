"""경량 2D 트래커 — 검출(2~5Hz) 사이의 프레임을 보간한다.

실시간 루프에서 매 프레임 검출은 불가능하다(YOLOE ~200ms). 검출은 가끔 하고, 그
사이는 이 트래커가 박스를 이어받아 화면을 30fps로 부드럽게 만든다(docs/REALTIME.md §2).

OpenCV의 KCF 트래커를 래핑한다. KCF는 빠르고(수 ms) 짧은 구간 보간에 충분하다.
가림/급격한 움직임엔 약하지만, 그건 3D 근접 연관(object_map)이 보완한다.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..utils.logging import get_logger

log = get_logger(__name__)


class BoxTracker:
    """단일 물체의 bbox를 프레임 간 추적한다(검출 사이 보간용).

    검출이 새로 오면 :meth:`reinit` 로 재초기화하고, 그 사이 프레임은
    :meth:`update` 로 박스를 갱신한다.
    """

    def __init__(self) -> None:
        self._tracker = None
        self._last_box: np.ndarray | None = None
        self._ok = False

    def reinit(self, rgb: np.ndarray, bbox_xyxy: np.ndarray) -> None:
        """새 검출 박스로 트래커를 (재)초기화한다."""
        x1, y1, x2, y2 = bbox_xyxy
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        # OpenCV 트래커는 (x, y, w, h) 형식.
        self._tracker = cv2.TrackerKCF_create()
        self._tracker.init(_to_bgr(rgb), (int(x1), int(y1), int(w), int(h)))
        self._last_box = np.asarray(bbox_xyxy, dtype=np.float64)
        self._ok = True

    def update(self, rgb: np.ndarray) -> np.ndarray | None:
        """다음 프레임에서 박스를 갱신한다. 추적 실패면 None."""
        if self._tracker is None:
            return None
        ok, box = self._tracker.update(_to_bgr(rgb))
        if not ok:
            self._ok = False
            return None
        x, y, w, h = box
        self._last_box = np.array([x, y, x + w, y + h], dtype=np.float64)
        return self._last_box


def _to_bgr(rgb: np.ndarray) -> np.ndarray:
    """OpenCV 트래커는 BGR을 기대한다."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
