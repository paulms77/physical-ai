"""iOS ↔ 파이썬 프로토콜 계약 검증.

Swift(FrameEncoder.swift)를 이 리눅스 서버에서 못 돌리므로, **Swift가 만들어낼
바이트를 파이썬으로 그대로 재현**해서 decode_frame이 올바로 푸는지 확인한다.
이게 두 독립 구현(Swift/파이썬)의 계약을 못박는 안전망이다.

특히 두 지점이 깨지기 쉽다:
1. 행렬 전치 — ARKit simd_float4x4는 열 우선, 프로토콜은 행 우선.
2. depth 양자화 — float32 미터 → uint16 밀리미터 리틀엔디언.

실행:  .venv/bin/python tests/test_ios_protocol_contract.py
"""

from __future__ import annotations

import json
import struct

import numpy as np

from pysical.io.protocol import decode_frame


def _swift_style_encode_pose_only(timestamp, T_row_major_flat, intrinsics):
    """Swift FrameEncoder.encode가 (RGB/depth 없이) 만들 바이트를 재현.

    Swift는 헤더 JSON + [4바이트 BE 길이] 접두. RGB 없으면 페이로드 없음.
    """
    header = {
        "timestamp": timestamp,
        "T_world_camera": T_row_major_flat,  # Swift가 rowMajor()로 편 것
        "tracking_state": "normal",
        "has_rgb": False,
        "intrinsics": intrinsics,
    }
    hb = json.dumps(header).encode("utf-8")
    return struct.pack(">I", len(hb)) + hb


def test_row_major_pose_decodes_correctly():
    """Swift rowMajor()로 편 포즈가 파이썬에서 원래 4x4로 복원된다."""
    # 알려진 포즈: 위치 (1,2,3), 회전은 단위
    T = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 2],
        [0, 0, 1, 3],
        [0, 0, 0, 1],
    ], dtype=np.float64)
    # Swift rowMajor(m)는 행 우선으로 편다 → T.flatten(order='C')와 같아야
    row_major = T.flatten(order="C").tolist()

    msg = _swift_style_encode_pose_only(1.5, row_major, None)
    frame = decode_frame(msg)

    assert np.allclose(frame.T_world_camera, T), frame.T_world_camera
    # 위치가 올바로 복원됐는지(전치 실수면 여기서 깨진다)
    assert np.allclose(frame.camera_position_world, [1, 2, 3])


def test_column_major_would_fail():
    """(반례) 열 우선으로 잘못 보내면 포즈가 뒤집힌다 — 계약 위반 감지."""
    T = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 2],
        [0, 0, 1, 3],
        [0, 0, 0, 1],
    ], dtype=np.float64)
    wrong = T.flatten(order="F").tolist()  # 열 우선(전치 안 함)
    frame = decode_frame(_swift_style_encode_pose_only(0.0, wrong, None))
    # 위치가 (1,2,3)이 아니게 나와야 정상(= 잘못된 방식임을 확인)
    assert not np.allclose(frame.camera_position_world, [1, 2, 3])


def test_depth_uint16_le_quantization_matches():
    """Swift quantizeDepth(uint16 mm LE)를 재현해 decode가 미터로 복원하는지."""
    # 2x3 depth (미터). 0은 무효.
    depth_m = np.array([[0.5, 1.0, 0.0], [2.5, 0.0, 3.0]], dtype=np.float32)
    h, w = depth_m.shape
    # Swift: min(m*1000, 65535), 무효(0/비유한)는 0
    mm = np.zeros((h, w), dtype="<u2")
    for y in range(h):
        for x in range(w):
            v = depth_m[y, x]
            if np.isfinite(v) and v > 0:
                mm[y, x] = min(int(v * 1000), 65535)
    depth_bytes = mm.tobytes()

    intr = {"fx": 200.0, "fy": 200.0, "cx": 100.0, "cy": 75.0}
    header = {
        "timestamp": 0.0,
        "T_world_camera": np.eye(4).flatten(order="C").tolist(),
        "tracking_state": "normal",
        "has_rgb": False,
        "intrinsics": None,
        "depth_lidar": {"h": h, "w": w, "bytes": len(depth_bytes)},
        "depth_intrinsics": intr,
    }
    hb = json.dumps(header).encode("utf-8")
    msg = struct.pack(">I", len(hb)) + hb + depth_bytes

    frame = decode_frame(msg)
    assert frame.has_lidar_depth
    # 유효값은 mm→m로 복원(1mm 오차 이내)
    assert np.isclose(frame.depth_lidar[0, 0], 0.5, atol=1e-3)
    assert np.isclose(frame.depth_lidar[1, 2], 3.0, atol=1e-3)
    # 무효(0)는 NaN
    assert np.isnan(frame.depth_lidar[0, 2])
    assert np.isnan(frame.depth_lidar[1, 1])
    assert np.isclose(frame.depth_intrinsics.fx, 200.0)


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} 통과")
    return failed == 0


if __name__ == "__main__":
    import sys

    sys.exit(0 if _run_all() else 1)
