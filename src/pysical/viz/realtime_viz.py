"""실시간 시각화 — rerun에 카메라·검출박스·마스크·3D object map·Auto-Focus를 함께 띄운다.

실시간 루프의 매 프레임 결과를 rerun에 스트리밍한다. 세 가지 뷰가 함께 뜬다:

    1. 3D 뷰(map/)       — 카메라 궤적 + object map 인스턴스(고유 ID·world 좌표)
    2. 2D 뷰(camera/)    — RGB + 검출 박스/마스크 오버레이
    3. Focus 뷰(focus/)  — 선택 대상의 확대 정밀 뷰(Auto-Focus)

RerunVisualizer(카메라/궤적)를 재사용하고, 그 위에 인식 결과를 얹는다.
"""

from __future__ import annotations

import numpy as np
import rerun as rr

from ..io.types import Frame
from ..perception.types import Detection
from ..tracking.object_map import TrackedObject
from .rerun_viz import RerunVisualizer


class RealtimeVisualizer:
    """실시간 detect-and-track 결과를 rerun에 표시한다."""

    def __init__(
        self,
        app_id: str = "pysical_realtime",
        spawn_viewer: bool = True,
        save_path: str | None = None,
        serve_web: bool = False,
    ) -> None:
        self._base = RerunVisualizer(
            app_id=app_id,
            spawn_viewer=spawn_viewer,
            save_path=save_path,
            serve_web=serve_web,
        )

    def log(
        self,
        frame: Frame,
        boxes: list[Detection],
        tracked: list[TrackedObject],
        focus_view: np.ndarray | None = None,
    ) -> None:
        """한 프레임의 전체 시각화를 기록한다."""
        # 1. 카메라 궤적 + (있으면) RGB — 기존 시각화 재사용.
        self._base.log_frame(frame)

        rr.set_time("iphone_clock", duration=frame.timestamp)

        # 2. 2D 검출 박스 오버레이 (RGB 뷰 위에).
        #
        # **비었을 때 반드시 지워야 한다.** rerun은 새 값이 올 때까지 마지막 값을
        # 계속 그린다. 검출이 없다고 로깅을 건너뛰면 대상이 사라진 자리에 마지막
        # 박스가 잔상으로 남는다(실측).
        if boxes:
            rr.log(
                "camera_image/detections",
                rr.Boxes2D(
                    array=np.array([_xyxy_to_xywh(b.bbox_xyxy) for b in boxes]),
                    array_format=rr.Box2DFormat.XYWH,
                    labels=[f"{b.label} {b.score:.2f}" for b in boxes],
                ),
            )
        else:
            rr.log("camera_image/detections", rr.Clear(recursive=True))

        # 3. 3D object map 인스턴스 — 고유 ID·world 좌표. 화면 밖 물체도 남는다.
        if tracked:
            positions = np.array([o.position_world for o in tracked])
            labels = [f"{o.label}#{o.obj_id}" for o in tracked]
            # 유예 중(안 보이는) 인스턴스는 흐리게 표시.
            colors = [
                [255, 200, 0] if o.misses == 0 else [120, 120, 120] for o in tracked
            ]
            rr.log(
                "map/objects",
                rr.Points3D(positions, radii=0.03, labels=labels, colors=colors),
            )
        else:
            rr.log("map/objects", rr.Clear(recursive=True))

        # 4. Auto-Focus 확대 뷰(별도 2D 패널). 대상이 없으면 패널도 비운다.
        if focus_view is not None:
            rr.log("focus", rr.Image(focus_view))
        else:
            rr.log("focus", rr.Clear(recursive=True))


def _xyxy_to_xywh(bbox_xyxy: np.ndarray) -> list[float]:
    """(x1,y1,x2,y2) → (x, y, w, h). rerun Boxes2D XYWH 형식용."""
    x1, y1, x2, y2 = bbox_xyxy
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
