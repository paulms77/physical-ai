"""3D 근접 기반 인스턴스 연관 (object map).

실시간의 강건함이 여기서 나온다(docs/REALTIME.md §3). 각 물체 인스턴스에 고유 ID와
world 좌표를 부여하고, 프레임 간에 **3D 위치로** 같은 물체인지 판단한다.

왜 2D 트래킹보다 나은가
----------------------
2D 트래커는 물체가 가려지거나 화면 밖으로 나가면 잃는다. 하지만 우리는 이미 world
좌표가 있다 — 물체가 잠깐 안 보여도 **물리적 위치는 그대로**다. 직전 프레임에
(0.43, -0.62, -1.14)에 있던 컵과 이번 검출이 수 cm 안이면 같은 컵으로 본다.

부수 효과로 "이 공간에 무엇이 어디 있는지" 아는 작은 object map이 생긴다. 화면 밖으로
나간 물체도 목록에 남고, 다시 보이면 새 물체가 아니라 같은 물체(같은 ID)로 인식된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class TrackedObject:
    """object map이 관리하는 한 인스턴스.

    Attributes
    ----------
    obj_id:
        고유 ID(프레임이 바뀌어도 유지).
    label:
        클래스 이름.
    position_world:
        마지막으로 확인된 world 좌표(3,).
    score:
        마지막 검출 신뢰도.
    last_seen_frame:
        마지막으로 관측된 프레임 번호. 오래 안 보이면 제거 판단에 쓴다.
    misses:
        연속으로 관측 안 된 횟수(유예 기간 판정용).
    """

    obj_id: int
    label: str
    position_world: np.ndarray
    score: float
    last_seen_frame: int
    misses: int = 0


class ObjectMap:
    """world 좌표 기반으로 인스턴스를 연관·유지하는 맵.

    Parameters
    ----------
    assoc_dist_m:
        이 거리(미터) 안이면 같은 인스턴스로 본다. 기본 8cm.
    max_misses:
        이만큼 연속으로 관측 안 되면 맵에서 제거(유예 기간). 손 가림/짧은 이탈은
        이 유예로 버틴다.
    """

    def __init__(self, assoc_dist_m: float = 0.08, max_misses: int = 8) -> None:
        self._objects: dict[int, TrackedObject] = {}
        self._next_id = 0
        self._assoc_dist = assoc_dist_m
        self._max_misses = max_misses

    def update(self, frame_idx: int, observations: list[tuple[str, np.ndarray, float]]) -> list[TrackedObject]:
        """이번 프레임의 관측들로 맵을 갱신한다.

        Parameters
        ----------
        frame_idx:
            현재 프레임 번호.
        observations:
            (label, position_world(3,), score) 리스트. 이번에 검출·추정된 물체들.

        Returns
        -------
        list[TrackedObject]
            현재 살아있는(유예 포함) 인스턴스들.
        """
        matched_ids: set[int] = set()

        for label, pos, score in observations:
            pos = np.asarray(pos, dtype=np.float64)
            obj_id = self._find_match(label, pos, matched_ids)
            if obj_id is not None:
                # 기존 인스턴스 갱신
                o = self._objects[obj_id]
                o.position_world = pos
                o.score = score
                o.last_seen_frame = frame_idx
                o.misses = 0
                matched_ids.add(obj_id)
            else:
                # 새 인스턴스 등록
                o = TrackedObject(self._next_id, label, pos, score, frame_idx)
                self._objects[self._next_id] = o
                matched_ids.add(self._next_id)
                self._next_id += 1

        # 이번에 매칭 안 된 기존 인스턴스는 miss 카운트↑, 유예 넘으면 제거.
        for obj_id in list(self._objects.keys()):
            if obj_id not in matched_ids:
                self._objects[obj_id].misses += 1
                if self._objects[obj_id].misses > self._max_misses:
                    del self._objects[obj_id]

        return list(self._objects.values())

    def _find_match(self, label: str, pos: np.ndarray, taken: set[int]) -> int | None:
        """같은 label 중 pos에 가장 가깝고 임계 안인 인스턴스 ID를 찾는다."""
        best_id, best_dist = None, self._assoc_dist
        for obj_id, o in self._objects.items():
            if obj_id in taken or o.label != label:
                continue
            d = float(np.linalg.norm(o.position_world - pos))
            if d < best_dist:
                best_id, best_dist = obj_id, d
        return best_id

    def keep_only_labels(self, labels: set[str]) -> None:
        """대상 어휘에 없는 인스턴스를 버린다.

        사용자가 "컵 말고 펜"이라고 하면 컵 인스턴스는 더 이상 추적 대상이 아니다.
        그대로 두면 3D 뷰에 컵 좌표가 영구히 남는다(실측).
        """
        for obj_id in [i for i, o in self._objects.items() if o.label not in labels]:
            del self._objects[obj_id]

    def clear(self) -> None:
        """모든 인스턴스를 버린다."""
        self._objects.clear()

    @property
    def objects(self) -> list[TrackedObject]:
        """현재 맵의 모든 인스턴스(유예 중 포함)."""
        return list(self._objects.values())

    @property
    def visible_objects(self) -> list[TrackedObject]:
        """이번 프레임에 실제로 보인(miss=0) 인스턴스들."""
        return [o for o in self._objects.values() if o.misses == 0]
