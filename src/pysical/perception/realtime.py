"""실시간 인식 엔진 — 검출 주기 제어 + 트래커 보간 + 3D object map 통합.

docs/REALTIME.md의 detect-and-track 구조를 코드로 옮긴 곳이다. 매 프레임 검출은
불가능하므로(YOLOE ~200ms), 검출은 2~5Hz로만 하고 그 사이는 트래커가 박스를
이어받는다. 3D world 좌표는 object map이 인스턴스별로 유지한다(손 가림에 강건).

한 프레임 처리 흐름
------------------
    process(frame):
        if 검출할 차례:  YOLOE 검출 → 트래커 재초기화 → 3D 위치추정 → object map 갱신
        else:            트래커로 박스만 보간 (빠름) → object map은 유예 유지
        return 현재 인스턴스들(TrackedObject) + 표시용 박스

이 엔진은 입력이 실시간 스트림이든 녹화 재생이든 동일하게 동작한다(같은 Frame).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from ..io.types import Frame
from ..tracking.object_map import ObjectMap, TrackedObject
from ..tracking.tracker import BoxTracker
from ..utils.logging import get_logger
from .localize import localize_object
from .types import Detection

log = get_logger(__name__)


@dataclass
class RealtimeConfig:
    """실시간 엔진 설정."""

    detect_every: int = 5        # N 프레임마다 검출(나머지는 트래커). 30fps면 6Hz.
    localize: bool = True        # depth로 3D 위치추정 할지(있을 때).
    assoc_dist_m: float = 0.08   # object map 3D 연관 임계.
    max_misses: int = 8          # 손 가림 유예(프레임).
    # --- 2D 박스 수명 관리(잔상 방지) ---
    # KCF는 대상이 화면을 벗어나도 배경에 눌어붙어 ok=True를 반환한다. 검출로
    # 확인되지 않은 보간 박스가 오래 남지 않도록 아래 둘로 직접 정리한다.
    min_visible_ratio: float = 0.35   # 화면 안에 남은 면적 비율이 이 아래면 버린다.
    max_track_age: int | None = None  # 마지막 검출 이후 보간 허용 프레임(기본 detect_every).


class RealtimeEngine:
    """프레임을 받아 검출·추적·3D 유지를 수행하는 실시간 엔진.

    Parameters
    ----------
    detector:
        BaseDetector(set_prompt로 대상 어휘 고정됨). None이면 검출 없이 트래킹만.
    config:
        주기·유예 등 설정.
    """

    def __init__(self, detector, config: RealtimeConfig | None = None) -> None:
        self._detector = detector
        self._cfg = config or RealtimeConfig()
        self._object_map = ObjectMap(self._cfg.assoc_dist_m, self._cfg.max_misses)
        # 검출된 물체마다 하나의 트래커. (라벨+인덱스)로 관리.
        self._trackers: list[tuple[str, BoxTracker]] = []
        self._frame_idx = -1
        self._frames_since_detect = 0
        # 대상 어휘가 아직 없으면 검출을 돌리지 않는다(음성으로 지정할 때까지 대기).
        self._has_prompt = False
        self._labels: list[str] = []
        # 검출 스레드와 표시 스레드가 트래커/맵을 함께 만지므로 보호한다.
        # 무거운 추론 자체는 락 밖에서 돌려 표시 루프를 막지 않는다.
        self._lock = threading.Lock()
        # 검출기 모델 자체를 보호한다. set_prompt(클래스 교체)가 detect(추론)와
        # 겹치면 같은 torch 모델을 두 스레드가 동시에 만져 예외가 난다.
        # _lock과 절대 중첩해 잡지 않는다(교착 방지).
        self._model_lock = threading.Lock()
        self._last_boxes: list[Detection] = []

    def set_prompt(self, labels: list[str]) -> None:
        """추적 대상을 바꾼다(사용자가 새 명령을 줄 때). 즉시 반영.

        빈 리스트를 주면 대상을 해제한다. 그동안 검출은 돌지 않고 포즈·시각화만
        이어진다(음성으로 대상을 지정할 때까지 대기하는 상태).
        """
        # 대상이 바뀌면 기존 트래커/박스를 비운다.
        with self._lock:
            self._trackers.clear()
            self._last_boxes = []
            self._frames_since_detect = 0

        if not labels:
            with self._lock:
                self._has_prompt = False
                self._labels = []
                self._object_map.clear()   # 대상 없음 = 3D 뷰도 비운다
            return

        if self._detector is not None:
            # 진행 중인 추론이 끝나기를 기다렸다가 클래스를 교체한다.
            with self._model_lock:
                self._detector.set_prompt(labels)

        with self._lock:
            self._labels = list(labels)
            self._has_prompt = True
            # 새 대상에 없는 인스턴스는 버린다("컵 말고 펜" → 컵 좌표 제거).
            self._object_map.keep_only_labels(set(labels))

    def process(self, frame: Frame) -> tuple[list[Detection], list[TrackedObject]]:
        """한 프레임을 처리한다.

        Returns
        -------
        (display_boxes, tracked_objects)
            display_boxes: 이번 프레임에 화면에 그릴 박스들(검출 또는 트래커 보간).
            tracked_objects: object map의 현재 인스턴스들(3D 좌표 포함).
        """
        self._frame_idx += 1
        is_detect_frame = (
            self._detector is not None
            and self._has_prompt
            and self._frame_idx % self._cfg.detect_every == 0
            and frame.rgb is not None
        )

        if is_detect_frame:
            boxes = self._detect_and_reinit(frame)
            self._frames_since_detect = 0
        else:
            self._frames_since_detect += 1
            boxes = self._track_only(frame)

        tracked = self._update_map(frame, boxes) if is_detect_frame else self._object_map.objects
        self._last_boxes = boxes
        return boxes, tracked

    def _update_map(self, frame: Frame, boxes: list[Detection]) -> list[TrackedObject]:
        """검출 결과로 3D 위치추정 + object map을 갱신한다(depth 있을 때만)."""
        if not (self._cfg.localize and frame.has_any_depth):
            return self._object_map.objects
        observations: list[tuple[str, np.ndarray, float]] = []
        for det in boxes:
            locs = localize_object(det, frame)
            if locs:
                loc = next(iter(locs.values()))
                observations.append((det.label, loc.position_world, det.score))
        # 관측이 비어도 update를 부른다. 여기서 빠져나가면 misses가 올라가지 않아
        # 사라진 물체가 max_misses에 도달하지 못하고 3D 뷰에 영구히 남는다.
        return self._object_map.update(self._frame_idx, observations)

    # --- 병렬 파이프라인용 API -------------------------------------------------
    #
    # process()는 검출과 보간을 한 스레드에서 번갈아 한다. 그러면 검출이 도는
    # 프레임만 유독 오래 걸려(MPS 34ms vs 보간 5ms) 화면이 주기적으로 끊긴다.
    # 아래 둘로 나누면 검출 스레드가 자기 속도로 돌고, 표시 스레드는 일정한
    # 주기를 유지한다.

    def detect_step(self, frame: Frame) -> list[Detection]:
        """검출 스레드용. 무거운 추론을 돌리고 트래커·object map을 갱신한다."""
        if self._detector is None or not self._has_prompt or frame.rgb is None:
            return []
        # 모델 락만 잡는다(표시 루프의 _lock은 건드리지 않아 보간은 계속 돈다).
        with self._model_lock:
            detections = self._detector.detect(frame)
        with self._lock:
            self._trackers = []
            for det in detections:
                tr = BoxTracker()
                tr.reinit(frame.rgb, det.bbox_xyxy)
                self._trackers.append((det.label, tr))
            self._frames_since_detect = 0
            self._last_boxes = detections
            self._frame_idx += 1
            self._update_map(frame, detections)
        return detections

    def track_step(self, frame: Frame) -> tuple[list[Detection], list[TrackedObject]]:
        """표시 스레드용. 트래커 보간만 하고 현재 인스턴스를 함께 돌려준다."""
        with self._lock:
            self._frames_since_detect += 1
            boxes = self._track_only(frame)
            self._last_boxes = boxes
            return boxes, self._object_map.objects

    def _detect_and_reinit(self, frame: Frame) -> list[Detection]:
        """검출 프레임: YOLOE로 검출하고 트래커를 재초기화한다."""
        detections = self._detector.detect(frame)
        self._trackers = []
        for det in detections:
            tr = BoxTracker()
            tr.reinit(frame.rgb, det.bbox_xyxy)
            self._trackers.append((det.label, tr))
        return detections

    def _track_only(self, frame: Frame) -> list[Detection]:
        """비검출 프레임: 트래커로 박스만 보간한다(빠름).

        보간 박스는 '검출로 확인된 사실'이 아니라 추정이다. KCF는 대상이 화면
        밖으로 나가도 실패를 보고하지 않으므로(배경에서 상관 피크를 찾는다),
        수명과 화면 이탈을 여기서 직접 정리한다. SORT의 ``max_age``에 해당한다.
        """
        if self._is_stale():
            # 마지막 검출로부터 너무 오래됐다. 추정만으로 박스를 남기지 않는다.
            self._trackers.clear()
            self._last_boxes = []
            return []

        if frame.rgb is None:
            return self._last_boxes

        h, w = frame.rgb.shape[:2]
        alive: list[tuple[str, BoxTracker]] = []
        out: list[Detection] = []
        for label, tr in self._trackers:
            box = tr.update(frame.rgb)
            if box is None:
                continue  # 트래커가 실패를 보고한 경우
            if not _is_visible(box, w, h, self._cfg.min_visible_ratio):
                continue  # 화면을 벗어났다 — 트래커째로 버린다
            alive.append((label, tr))
            # 트래커 보간은 신뢰도를 원본보다 살짝 낮춰 표시(추정임을 반영).
            out.append(Detection(label=label, score=0.5, bbox_xyxy=box))
        self._trackers = alive
        return out

    def _is_stale(self) -> bool:
        """마지막 검출 이후 보간 허용 프레임 수를 넘겼는지."""
        max_age = self._cfg.max_track_age
        if max_age is None:
            max_age = self._cfg.detect_every
        return self._frames_since_detect > max_age

    @property
    def labels(self) -> list[str]:
        """현재 검출 중인 대상 어휘."""
        return list(self._labels)

    @property
    def object_map(self) -> ObjectMap:
        return self._object_map


def _is_visible(box: np.ndarray, width: int, height: int, min_ratio: float) -> bool:
    """박스가 아직 화면 안에 유효하게 남아 있는지.

    중심이 화면을 벗어났거나, 화면 안에 남은 면적 비율이 ``min_ratio`` 아래이거나,
    크기가 비정상(찌그러짐/폭주)이면 False. 카메라를 돌려 대상이 프레임 밖으로
    나갈 때 박스가 남는 것을 막는다.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = x2 - x1, y2 - y1
    if bw < 4.0 or bh < 4.0:
        return False                      # 찌그러진 박스
    if bw > width * 1.5 or bh > height * 1.5:
        return False                      # 폭주한 박스

    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if not (0.0 <= cx < width and 0.0 <= cy < height):
        return False                      # 중심이 화면 밖

    inside_w = max(0.0, min(float(width), x2) - max(0.0, x1))
    inside_h = max(0.0, min(float(height), y2) - max(0.0, y1))
    return (inside_w * inside_h) / (bw * bh) >= min_ratio
