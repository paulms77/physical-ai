"""녹화 저장·재생 왕복 테스트.

RecordingWriter로 쓴 .pyrec를 read_recording으로 읽어도 프레임이 보존되는지 확인한다.
이게 깨지면 ③단계 재생 검증을 신뢰할 수 없다.

실행:  .venv/bin/python tests/test_recording.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from pysical.geometry.mock_camera import orbit_pose_arkit
from pysical.io.recording import RecordingWriter, read_recording
from pysical.io.types import CameraIntrinsics, Frame


def _make_frames(n=5):
    frames = []
    for i in range(n):
        t = i * 0.1
        rgb = np.full((32, 32, 3), i * 10, dtype=np.uint8)
        depth = np.full((24, 32), 1.0 + i * 0.1, dtype=np.float32)
        frames.append(
            Frame(
                timestamp=t,
                T_world_camera=orbit_pose_arkit(t),
                rgb=rgb,
                intrinsics=CameraIntrinsics(150, 150, 16, 16),
                depth_lidar=depth,
                depth_intrinsics=CameraIntrinsics(120, 120, 16, 12),
            )
        )
    return frames


def test_recording_roundtrip():
    """쓴 프레임 수·타임스탬프·포즈·depth가 재생에서 보존된다."""
    frames = _make_frames(5)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.pyrec"
        with RecordingWriter(path) as w:
            for f in frames:
                w.write(f)

        replayed = list(read_recording(path))

    assert len(replayed) == 5
    for orig, rep in zip(frames, replayed):
        assert np.isclose(rep.timestamp, orig.timestamp)
        assert np.allclose(rep.T_world_camera, orig.T_world_camera)
        assert rep.has_lidar_depth
        assert rep.rgb is not None


def test_recording_preserves_order():
    """프레임 순서가 유지된다(시계열 재생에 필수)."""
    frames = _make_frames(4)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "o.pyrec"
        with RecordingWriter(path) as w:
            for f in frames:
                w.write(f)
        ts = [f.timestamp for f in read_recording(path)]
    assert ts == sorted(ts)


def test_empty_recording():
    """빈 녹화는 0 프레임을 낸다(에러 없이)."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "empty.pyrec"
        with RecordingWriter(path):
            pass
        assert list(read_recording(path)) == []


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
