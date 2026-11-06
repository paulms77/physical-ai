"""rerun 기반 3D 시각화.

Phase 0의 완료 조건은 "폰을 들고 걸었을 때 화면 속 카메라가 따라 움직이는 것을
보는 것"이다(README §10). 이 모듈이 바로 그 '화면 속 카메라'를 그린다.

rerun은 좌표계를 명시할 수 있다. 우리는 받은 ARKit 포즈를 ROS 규약으로 변환한
뒤(:func:`arkit_camera_pose_to_ros`) 그린다. 그래야 이 시각화가 Phase 2의
RViz와 같은 축을 쓰게 되어, "여기서 위로 가면 RViz에서도 위"가 보장된다.

좌표축 색 관례(ROS/rerun 공통): X=빨강, Y=초록, Z=파랑.
ROS map에서 Z=위이므로, 카메라를 위로 들면 파란 축을 따라 위로 올라가야 한다.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import rerun as rr

from ..geometry.transforms import arkit_camera_pose_to_ros
from ..io.types import Frame
from ..utils.logging import get_logger

log = get_logger(__name__)


class RerunVisualizer:
    """카메라 포즈와 궤적을 rerun에 스트리밍한다.

    Parameters
    ----------
    app_id:
        rerun 세션 이름.
    spawn_viewer:
        True면 뷰어 창을 자동으로 띄운다(헤드리스 서버면 False로 두고 .rrd 저장).
    trail_length:
        궤적으로 남길 최근 카메라 위치 개수.
    axis_length_m:
        각 카메라 위치에 그리는 좌표축 길이(미터).
    """

    def __init__(
        self,
        app_id: str = "pysical_phase0",
        spawn_viewer: bool = True,
        trail_length: int = 200,
        axis_length_m: float = 0.1,
        save_path: str | None = None,
        serve_web: bool = False,
    ) -> None:
        """rerun 시각화를 초기화한다.

        시각화 출력 모드(하나만 고른다):
        - ``spawn_viewer=True`` : 로컬 GUI 뷰어 창을 띄운다(화면 있는 노트북용).
        - ``save_path=...``     : ``.rrd`` 파일로 저장(헤드리스 서버 → 나중에 로컬에서 열람).
        - ``serve_web=True``    : 웹 뷰어를 띄운다(헤드리스 서버 → 브라우저로 접속).

        헤드리스 SSH 서버에서 spawn_viewer=True면 "Rerun Viewer executable 없음"
        에러가 난다. 그럴 땐 save_path나 serve_web을 쓴다.
        """
        self._trail: deque[np.ndarray] = deque(maxlen=trail_length)
        self._axis_length = axis_length_m

        rr.init(app_id, spawn=(spawn_viewer and not save_path and not serve_web))
        if save_path:
            rr.save(save_path)
            log.info("rerun → 파일 저장: %s (나중에 `rerun %s`로 열람)", save_path, save_path)
        elif serve_web:
            # 브라우저로 접속하는 웹 뷰어 + gRPC 서버. 헤드리스 서버에 적합.
            rr.serve_web()
            log.info("rerun 웹 뷰어 시작 — 브라우저로 접속하세요(로그 URL 참고)")
        # 월드 좌표계가 Z-up(ROS map)임을 뷰어에 알려, 카메라 시점을 자연스럽게 잡게 한다.
        rr.log("map", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        # 레이아웃을 코드에서 지정한다. 지정하지 않으면 뷰어가 이전 세션의 레이아웃을
        # 기억해, 새로 추가된 뷰(카메라·Focus)가 화면에 나타나지 않는다(실측).
        _send_default_blueprint()
        log.info("rerun 초기화 완료 (app_id=%s)", app_id)

    def log_frame(self, frame: Frame) -> None:
        """한 프레임의 카메라 포즈·궤적·(있으면)RGB를 rerun에 기록한다."""
        # 1) ARKit 포즈 → ROS map 규약 포즈로 변환.
        T_map_cam = arkit_camera_pose_to_ros(frame.T_world_camera)
        position = T_map_cam[:3, 3]
        rotation = T_map_cam[:3, :3]

        # rerun 시간축을 iPhone 타임스탬프에 맞춘다(프레임 순서·재생용).
        # rerun 0.37: set_time(timeline, duration=<초>) 로 경과 시간축을 만든다.
        rr.set_time("iphone_clock", duration=frame.timestamp)

        # 2) 카메라를 3D 변환(Transform3D)으로 기록 → 뷰어가 좌표축을 그려준다.
        rr.log(
            "map/camera",
            rr.Transform3D(translation=position, mat3x3=rotation),
        )
        # 좌표축 길이를 명시적으로 제어하려면 Arrows3D로 X(빨강)/Y(초록)/Z(파랑)를 그린다.
        L = self._axis_length
        rr.log(
            "map/camera/axes",
            rr.Arrows3D(
                origins=np.zeros((3, 3)),
                vectors=np.eye(3) * L,
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )

        # 3) 궤적: 지금까지의 카메라 위치를 선으로 잇는다.
        self._trail.append(position.copy())
        if len(self._trail) >= 2:
            rr.log(
                "map/camera_trail",
                rr.LineStrips3D([np.array(self._trail)], colors=[0xFFD700FF]),
            )

        # 4) tracking이 흔들리면(limited) 눈에 띄게 표시 — 실패 모드를 정직하게 보여준다.
        if not frame.is_tracking_ok:
            rr.log(
                "map/camera/tracking_warning",
                rr.TextLog(f"tracking={frame.tracking_state}", level=rr.TextLogLevel.WARN),
            )

        # 5) RGB가 있으면 별도 2D 뷰에 띄운다.
        if frame.rgb is not None:
            # 원본 RGB를 그대로 보내면 640x480x3 ≈ 0.9MB/프레임이라 15fps에서 뷰어로
            # 가는 대역폭이 병목이 된다. JPEG로 줄여 보낸다(화면 확인용이라 충분하다).
            rr.log("camera_image", rr.Image(frame.rgb).compress(jpeg_quality=70))


def _send_default_blueprint() -> None:
    """3D 맵 · 카메라 화면 · Focus 패널이 항상 함께 보이도록 레이아웃을 보낸다.

    엔티티 경로는 이 모듈과 realtime_viz가 기록하는 것과 일치해야 한다::

        map           3D 맵(카메라 궤적 + object map)
        camera_image  실제 카메라 RGB (+ /detections 박스)
        focus         Auto-Focus 확대 패널

    ``make_default=True``라 사용자가 뷰어에서 레이아웃을 바꾸면 그쪽이 우선한다.
    """
    import rerun.blueprint as rrb

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/map", name="3D 맵"),
            rrb.Vertical(
                rrb.Spatial2DView(origin="/camera_image", name="카메라"),
                rrb.Spatial2DView(origin="/focus", name="Focus"),
            ),
            column_shares=[2, 1],
        ),
        collapse_panels=False,
    )
    rr.send_blueprint(blueprint, make_active=True, make_default=True)
