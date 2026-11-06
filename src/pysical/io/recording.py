"""녹화 데이터 저장·재생 (③단계 내부 검증).

iOS 앱으로 실기기 라이브를 하기 전에, iPhone이 녹화한 **동영상 + 카메라 포즈 +
depth**를 프레임 단위로 흘려보내 전체 파이프라인을 시계열로 검증한다. "동영상만"이
아니라 각 프레임의 포즈·depth가 함께 있어야 3D 추정이 성립한다(그래서 iOS 앱의 첫
기능이 '녹화'다 — README §8).

포맷
----
WebSocket 와이어 프로토콜(:func:`encode_frame`)을 그대로 재활용한다. 각 프레임을
직렬화해 [4바이트 길이][프레임 바이트] 로 파일에 순차 기록한다(`.pyrec`). 장점:

- 실기기 라이브(WebSocket)와 **완전히 같은 코드 경로**(decode_frame)를 탄다.
  재생에서 되는 게 라이브에서도 된다.
- RGB(JPEG)·depth(uint16)·포즈·intrinsics가 한 프레임에 묶여 시간 동기화가 보장된다.

이 모듈은 저장(RecordingWriter)과 재생(read_recording)을 제공한다. iOS 앱이 아직
없으므로, mock 송신기로 만든 프레임을 녹화해 재생기를 자체 검증할 수도 있다.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterator

from ..utils.logging import get_logger
from .protocol import decode_frame, encode_frame
from .types import Frame

log = get_logger(__name__)

_LEN_FMT = ">I"
_LEN_SIZE = struct.calcsize(_LEN_FMT)
RECORDING_SUFFIX = ".pyrec"


class RecordingWriter:
    """프레임을 .pyrec 파일에 순차 기록한다.

    컨텍스트 매니저로 쓴다::

        with RecordingWriter("data/recordings/desk.pyrec") as w:
            for frame in frames:
                w.write(frame)
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._f = None
        self._count = 0

    def __enter__(self) -> "RecordingWriter":
        self._f = self._path.open("wb")
        log.info("녹화 시작: %s", self._path)
        return self

    def write(self, frame: Frame) -> None:
        if self._f is None:
            raise RuntimeError("with 블록 안에서 사용하세요.")
        payload = encode_frame(frame)
        self._f.write(struct.pack(_LEN_FMT, len(payload)))
        self._f.write(payload)
        self._count += 1

    def __exit__(self, *exc) -> None:
        if self._f is not None:
            self._f.close()
        log.info("녹화 종료: %s (%d 프레임)", self._path, self._count)


def read_recording(path: str | Path) -> Iterator[Frame]:
    """.pyrec 파일을 프레임 단위로 순차 읽어 yield한다.

    제너레이터라 큰 녹화도 메모리에 다 올리지 않는다. 각 프레임은 실기기가 보낸 것과
    동일한 decode_frame으로 복원되므로, 파이프라인은 라이브인지 재생인지 알 필요가 없다.
    """
    path = Path(path)
    with path.open("rb") as f:
        while True:
            head = f.read(_LEN_SIZE)
            if len(head) < _LEN_SIZE:
                break  # 파일 끝
            (n,) = struct.unpack(_LEN_FMT, head)
            payload = f.read(n)
            if len(payload) < n:
                log.warning("녹화 파일이 중간에 잘렸습니다. 남은 프레임 무시.")
                break
            yield decode_frame(payload)
