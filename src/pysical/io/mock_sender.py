"""mock 프레임 송신기 (iPhone 역할을 대신하는 테스트 클라이언트).

수신기·좌표변환·시각화 전체를 iPhone 없이 검증하기 위해, 가상 카메라 포즈를
:mod:`pysical.geometry.mock_camera` 로 만들어 실제 iOS 앱과 **똑같은 와이어
프로토콜**(:func:`encode_frame`)로 WebSocket에 흘려보낸다.

사용 예::

    asyncio.run(run_mock_sender("ws://localhost:8765", fps=15, duration_s=20))
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import websockets

from ..geometry.mock_camera import orbit_pose_arkit
from ..utils.logging import get_logger
from .protocol import encode_frame
from .types import CameraIntrinsics, Frame

log = get_logger(__name__)

# 640x480 카메라의 그럴듯한 intrinsics (실제 값은 iPhone이 보내준다).
_MOCK_INTRINSICS = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0)


def _make_mock_rgb(t: float) -> np.ndarray:
    """시간에 따라 색이 변하는 더미 RGB 프레임. 실제 영상 대신 파이프라인 확인용."""
    h, w = 120, 160
    img = np.zeros((h, w, 3), dtype=np.uint8)
    hue = int((t * 30) % 255)
    img[:, :, 0] = hue
    img[:, :, 1] = 255 - hue
    img[:, :, 2] = 128
    return img


async def run_mock_sender(
    uri: str,
    fps: float = 15.0,
    duration_s: float = 20.0,
    radius_m: float = 1.0,
    height_m: float = 1.2,
    send_rgb: bool = True,
) -> None:
    """수신기에 접속해 가상 카메라 프레임을 주기적으로 보낸다.

    Parameters
    ----------
    uri:
        수신기 주소, 예: ``ws://localhost:8765``.
    fps:
        초당 프레임 수.
    duration_s:
        총 송신 시간. 0이면 무한.
    radius_m, height_m:
        가상 카메라의 원 궤적 반지름·높이.
    send_rgb:
        더미 RGB를 함께 보낼지.
    """
    interval = 1.0 / fps
    log.info("mock 송신기 접속 시도: %s", uri)
    async with websockets.connect(uri) as ws:
        log.info("접속 성공 — %.0f fps 로 송신 시작 (%.0fs)", fps, duration_s)
        start = time.monotonic()
        sent = 0
        while True:
            t = time.monotonic() - start
            if duration_s > 0 and t >= duration_s:
                break

            pose = orbit_pose_arkit(t, radius=radius_m, height=height_m)
            frame = Frame(
                timestamp=t,
                T_world_camera=pose,
                rgb=_make_mock_rgb(t) if send_rgb else None,
                intrinsics=_MOCK_INTRINSICS if send_rgb else None,
                tracking_state="normal",
            )
            await ws.send(encode_frame(frame))
            sent += 1
            await asyncio.sleep(interval)

        log.info("송신 완료 — 총 %d 프레임", sent)
