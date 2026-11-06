"""Auto-Focus 대상 선택 — 한번 잡은 대상을 놓지 않는다(lock-on).

기존에는 매 프레임 ``max(boxes, key=score)``로 대상을 다시 골랐다. 컵이 둘이면
신뢰도가 프레임마다 미세하게 뒤집히면서 포커스 패널이 두 컵 사이를 깜빡였다.
병변을 보여주는 의료 워크플로의 기능 증명이 목표이므로(viz/focus.py 참고), 대상이
저절로 바뀌는 것은 기능 자체의 결함이다.

동작
----
1. **고정(lock-on)** — 한번 고른 대상은 화면에서 사라지기 전까지 유지한다. 다른
   인스턴스의 신뢰도가 더 높아져도 바꾸지 않는다.
2. **추종** — 프레임마다 직전 위치에 가장 가까운 같은 라벨 박스를 이어받는다.
   검출 프레임과 트래커 보간 프레임 모두에서 동작한다(3D depth 불필요).
3. **유예** — 가려져 잠깐 사라지면 ``lost_grace`` 프레임 동안 기다린다. 그 안에
   다시 나타나면 같은 대상으로 이어지고, 넘기면 재선택한다.
4. **최초 선택** — 화면 중앙 근접도를 가장 크게 본다(사용자가 겨눈 것이 중앙에
   온다). 거리와 신뢰도를 보조로 쓴다.
5. **음성 전환** — "다음"은 인스턴스 순환, "가까운 거"는 최근접으로 즉시 전환.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.types import Frame
from ..perception.types import Detection


@dataclass
class FocusConfig:
    """대상 선택 가중치와 유예 설정."""

    center_weight: float = 1.0    # 화면 중앙 근접도(가장 크게 본다)
    near_weight: float = 0.5      # 카메라와의 거리(가까울수록 가점)
    score_weight: float = 0.35    # 검출 신뢰도(오검출 배제용 보조)
    lost_grace: int = 10          # 이 프레임 수까지는 사라져도 대상을 놓지 않는다
    match_ratio: float = 0.25     # 직전 위치와의 최대 매칭 거리(화면 대각 대비)
    near_range_m: float = 3.0     # 거리 점수를 0으로 떨어뜨리는 기준 거리


def _center(det: Detection) -> np.ndarray:
    x1, y1, x2, y2 = det.bbox_xyxy
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)


def _area(det: Detection) -> float:
    x1, y1, x2, y2 = det.bbox_xyxy
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def median_depth(frame: Frame, det: Detection) -> float | None:
    """대상 중심 부근의 depth 중앙값(미터). depth가 없으면 None.

    depth map은 RGB와 해상도가 다를 수 있어(LiDAR는 보통 256x192) 좌표를 비율로
    환산해 샘플링한다.
    """
    if frame.rgb is None:
        return None
    h, w = frame.rgb.shape[:2]
    if h == 0 or w == 0:
        return None

    for depth in (frame.depth_lidar, frame.depth_mono):
        if depth is None:
            continue
        dh, dw = depth.shape[:2]
        cx, cy = _center(det)
        u = int(cx * dw / w)
        v = int(cy * dh / h)
        if not (0 <= u < dw and 0 <= v < dh):
            continue
        win = depth[max(0, v - 3): v + 4, max(0, u - 3): u + 4]
        valid = win[np.isfinite(win) & (win > 0)]
        if valid.size:
            return float(np.median(valid))
    return None


class FocusSelector:
    """포커스 대상을 고르고 유지한다."""

    def __init__(self, config: FocusConfig | None = None) -> None:
        self._cfg = config or FocusConfig()
        self._label: str | None = None
        self._last_center: np.ndarray | None = None
        self._misses = 0
        self._pending: str | None = None
        # 라벨 지정 포커스("포커스 펜"). 아직 화면에 없으면 잠시 기다린다.
        self._pending_label: str | None = None
        self._label_retries = 0

    # --- 외부 명령 ---

    def request(self, select: str) -> None:
        """음성 전환 요청을 예약한다("next" | "nearest"). 다음 select에서 반영."""
        self._pending = select

    def request_label(self, label: str, retries: int = 45) -> None:
        """특정 라벨을 포커스 대상으로 지정한다("포커스 펜").

        검출 어휘는 그대로 두고 포커스만 그 종류로 옮긴다. 아직 화면에 없으면
        ``retries`` 프레임 동안 기다렸다가 포기한다(대상이 시야 밖일 수 있다).
        """
        self._pending_label = label
        self._label_retries = retries

    def reset(self) -> None:
        """고정을 해제한다(대상 어휘가 바뀌었을 때 등)."""
        self._label = None
        self._last_center = None
        self._misses = 0
        self._pending = None
        self._pending_label = None
        self._label_retries = 0

    @property
    def locked_label(self) -> str | None:
        return self._label

    # --- 매 프레임 호출 ---

    def select(self, boxes: list[Detection], frame: Frame) -> Detection | None:
        """이번 프레임의 포커스 대상을 돌려준다. 없으면 None."""
        if not boxes or frame.rgb is None:
            self._misses += 1
            if self._misses > self._cfg.lost_grace:
                self.reset()
            return None

        pending, self._pending = self._pending, None

        # 라벨 지정이 대기 중이면 그것부터. 아직 안 보이면 몇 프레임 기다린다.
        if self._pending_label is not None:
            wanted = [b for b in boxes if b.label == self._pending_label]
            if wanted:
                self._pending_label = None
                self._label_retries = 0
                target = self._pick_best(wanted, frame)
                self._label = target.label
                self._last_center = _center(target)
                self._misses = 0
                return target
            self._label_retries -= 1
            if self._label_retries <= 0:
                self._pending_label = None

        if pending == "nearest":
            target = self._pick_nearest(boxes, frame)
        elif pending == "next":
            target = self._pick_next(boxes)
        else:
            target = self._follow(boxes, frame)

        if target is not None:
            self._label = target.label
            self._last_center = _center(target)
            self._misses = 0
        return target

    # --- 내부 ---

    def _follow(self, boxes: list[Detection], frame: Frame) -> Detection | None:
        """고정된 대상을 이어받는다. 고정이 없거나 유예를 넘기면 새로 고른다."""
        if self._label is None or self._last_center is None:
            return self._pick_best(boxes, frame)

        h, w = frame.rgb.shape[:2]
        limit = float(np.hypot(w, h)) * self._cfg.match_ratio
        same = [b for b in boxes if b.label == self._label]
        if same:
            nearest = min(same, key=lambda b: float(np.linalg.norm(_center(b) - self._last_center)))
            if float(np.linalg.norm(_center(nearest) - self._last_center)) <= limit:
                return nearest

        # 대상을 놓쳤다. 유예 안이면 이번 프레임은 비우고 기다린다(깜빡임 방지).
        self._misses += 1
        if self._misses <= self._cfg.lost_grace:
            return None
        self.reset()
        return self._pick_best(boxes, frame)

    def _pick_best(self, boxes: list[Detection], frame: Frame) -> Detection:
        """중앙 근접도 > 거리 > 신뢰도 순으로 가중 평가해 고른다."""
        h, w = frame.rgb.shape[:2]
        img_center = np.array([w / 2.0, h / 2.0])
        half_diag = float(np.hypot(w, h)) / 2.0 or 1.0
        cfg = self._cfg

        def score(det: Detection) -> float:
            centrality = 1.0 - min(1.0, float(np.linalg.norm(_center(det) - img_center)) / half_diag)
            z = median_depth(frame, det)
            if z is not None:
                nearness = 1.0 - min(1.0, z / cfg.near_range_m)
            else:
                # depth가 없으면 박스 크기를 근접도의 대용으로 쓴다.
                nearness = min(1.0, _area(det) / float(w * h) * 4.0)
            return (
                cfg.center_weight * centrality
                + cfg.near_weight * nearness
                + cfg.score_weight * float(det.score)
            )

        return max(boxes, key=score)

    def _pick_nearest(self, boxes: list[Detection], frame: Frame) -> Detection:
        """카메라에서 가장 가까운 인스턴스. depth가 없으면 가장 큰 박스."""
        with_depth = [(median_depth(frame, b), b) for b in boxes]
        measured = [(z, b) for z, b in with_depth if z is not None]
        if measured:
            return min(measured, key=lambda zb: zb[0])[1]
        return max(boxes, key=_area)

    def _pick_next(self, boxes: list[Detection]) -> Detection:
        """같은 라벨 인스턴스를 좌→우 순서로 순환 전환한다."""
        # 같은 라벨 안에서 순환한다. 같은 라벨이 하나뿐이면(예: 컵 1개 + 펜 1개)
        # 그 안에서는 돌 곳이 없으므로 검출된 전체로 넓힌다.
        pool = [b for b in boxes if self._label is None or b.label == self._label]
        if len(pool) < 2:
            pool = boxes
        ordered = sorted(pool, key=lambda b: (_center(b)[0], _center(b)[1]))
        if self._last_center is None:
            return ordered[0]
        cur = min(
            range(len(ordered)),
            key=lambda i: float(np.linalg.norm(_center(ordered[i]) - self._last_center)),
        )
        return ordered[(cur + 1) % len(ordered)]
