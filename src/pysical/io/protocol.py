"""iPhone ↔ 노트북 WebSocket 와이어 프로토콜.

iOS 앱을 아직 만들지 않았으므로, **여기서 정한 포맷이 곧 앱이 따라야 할 계약**이다.
한 WebSocket 메시지 = 한 프레임이고, 다음 구조의 바이너리다::

    [4바이트 big-endian uint32 = JSON 헤더 길이 N]
    [N바이트 UTF-8 JSON 헤더]
    [나머지 전부 = JPEG로 인코딩된 RGB (헤더에 has_rgb=true일 때만)]

JSON 헤더 스키마::

    {
      "timestamp": 1234.567,          # iPhone 기준 초
      "T_world_camera": [16개 float],  # 4x4 행렬을 행 우선(row-major)으로 편 것
      "intrinsics": {"fx":.., "fy":.., "cx":.., "cy":..},  # 없으면 null
      "tracking_state": "normal",
      "has_rgb": true                  # 뒤에 JPEG가 붙는지
    }

이렇게 헤더(JSON)와 페이로드(JPEG)를 한 메시지에 함께 담으면, 포즈와 이미지가
'같은 프레임의 것'임이 자연스럽게 보장된다(README §11의 시간 동기화 문제 완화).
"""

from __future__ import annotations

import json
import struct

import cv2
import numpy as np

from .types import CameraIntrinsics, Frame

# 헤더 길이를 담는 접두 4바이트의 포맷 (big-endian unsigned int).
_HEADER_LEN_FMT = ">I"
_HEADER_LEN_SIZE = struct.calcsize(_HEADER_LEN_FMT)

# LiDAR depth 양자화 스케일. float32 미터를 uint16 밀리미터로 담아 대역폭을 절반으로
# 줄인다(README §5). 65535mm = 65.5m 까지 표현 — 실내 범위에 충분하다. 0은 '무효'.
_DEPTH_MM_SCALE = 1000.0


def encode_frame(frame: Frame, jpeg_quality: int = 80) -> bytes:
    """Frame을 와이어 바이너리로 직렬화한다 (주로 mock 송신기·테스트에서 사용).

    실제 데모에서는 iOS 앱이 이 포맷으로 직접 만들어 보내지만, 같은 규약을
    파이썬으로도 만들 수 있어야 mock 송신기와 왕복 테스트가 가능하다.
    """
    header: dict = {
        "timestamp": float(frame.timestamp),
        # 4x4 행렬을 행 우선으로 편다. iOS(simd_float4x4)와 맞추려면 앱에서도
        # 동일하게 행 우선으로 펴야 한다 — 이 규약을 문서에 못박아 둔다.
        "T_world_camera": frame.T_world_camera.flatten(order="C").tolist(),
        "tracking_state": frame.tracking_state,
        "has_rgb": frame.rgb is not None,
    }
    header["intrinsics"] = _intrinsics_to_dict(frame.intrinsics)

    # 페이로드는 여러 개(JPEG, depth, confidence)가 순서대로 붙는다. 각 조각의
    # 바이트 길이를 헤더에 적어, 수신 측이 정확히 잘라낼 수 있게 한다.
    payloads: list[bytes] = []

    if frame.rgb is not None:
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG 인코딩 실패")
        jpeg = buf.tobytes()
        header["rgb_bytes"] = len(jpeg)
        payloads.append(jpeg)

    # LiDAR depth: float32 미터 → uint16 밀리미터로 양자화(대역폭 절반).
    if frame.depth_lidar is not None:
        d = np.asarray(frame.depth_lidar, dtype=np.float32)
        d_mm = np.clip(d * _DEPTH_MM_SCALE, 0, 65535).astype("<u2")  # little-endian uint16
        depth_bytes = d_mm.tobytes()
        header["depth_lidar"] = {"h": d.shape[0], "w": d.shape[1], "bytes": len(depth_bytes)}
        header["depth_intrinsics"] = _intrinsics_to_dict(frame.depth_intrinsics)
        payloads.append(depth_bytes)

        if frame.depth_confidence is not None:
            c = np.asarray(frame.depth_confidence, dtype=np.uint8)
            conf_bytes = c.tobytes()
            header["depth_confidence"] = {"h": c.shape[0], "w": c.shape[1], "bytes": len(conf_bytes)}
            payloads.append(conf_bytes)

    header_bytes = json.dumps(header).encode("utf-8")
    parts = [struct.pack(_HEADER_LEN_FMT, len(header_bytes)), header_bytes, *payloads]
    return b"".join(parts)


def _intrinsics_to_dict(intr: CameraIntrinsics | None) -> dict | None:
    if intr is None:
        return None
    return {"fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy}


def _dict_to_intrinsics(d: dict | None) -> CameraIntrinsics | None:
    if not d:
        return None
    return CameraIntrinsics(d["fx"], d["fy"], d["cx"], d["cy"])


def decode_frame(message: bytes) -> Frame:
    """와이어 바이너리를 Frame으로 역직렬화한다 (수신기에서 사용).

    Raises
    ------
    ValueError
        메시지가 규약을 어겼을 때(길이 부족 등). 잘못된 프레임을 조용히 흘리지
        않고 여기서 막아, 원인 추적을 쉽게 한다.
    """
    if len(message) < _HEADER_LEN_SIZE:
        raise ValueError("메시지가 헤더 길이 접두어보다 짧습니다.")

    (header_len,) = struct.unpack(_HEADER_LEN_FMT, message[:_HEADER_LEN_SIZE])
    start = _HEADER_LEN_SIZE
    end = start + header_len
    if len(message) < end:
        raise ValueError("메시지가 헤더 길이만큼 담고 있지 않습니다.")

    header = json.loads(message[start:end].decode("utf-8"))

    T = np.array(header["T_world_camera"], dtype=np.float64).reshape(4, 4)
    intrinsics = _dict_to_intrinsics(header.get("intrinsics"))

    # 페이로드들을 헤더에 적힌 길이만큼 순서대로(RGB → depth → confidence) 잘라낸다.
    cursor = end

    def _take(n: int) -> bytes:
        nonlocal cursor
        chunk = message[cursor : cursor + n]
        if len(chunk) < n:
            raise ValueError("페이로드가 헤더에 명시된 길이보다 짧습니다.")
        cursor += n
        return chunk

    rgb = None
    if header.get("has_rgb"):
        jpeg = np.frombuffer(_take(header["rgb_bytes"]), dtype=np.uint8)
        bgr = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("JPEG 디코딩 실패")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    depth_lidar = None
    depth_confidence = None
    depth_intrinsics = None
    dmeta = header.get("depth_lidar")
    if dmeta:
        raw = _take(dmeta["bytes"])
        d_mm = np.frombuffer(raw, dtype="<u2").reshape(dmeta["h"], dmeta["w"])
        # uint16 mm → float32 미터. 0(무효)은 NaN으로 두어 뒤 단계에서 걸러지게 한다.
        depth_lidar = d_mm.astype(np.float32) / _DEPTH_MM_SCALE
        depth_lidar[d_mm == 0] = np.nan
        depth_intrinsics = _dict_to_intrinsics(header.get("depth_intrinsics"))

        cmeta = header.get("depth_confidence")
        if cmeta:
            craw = _take(cmeta["bytes"])
            depth_confidence = np.frombuffer(craw, dtype=np.uint8).reshape(cmeta["h"], cmeta["w"])

    return Frame(
        timestamp=header["timestamp"],
        T_world_camera=T,
        rgb=rgb,
        intrinsics=intrinsics,
        tracking_state=header.get("tracking_state", "normal"),
        depth_lidar=depth_lidar,
        depth_confidence=depth_confidence,
        depth_intrinsics=depth_intrinsics,
    )
