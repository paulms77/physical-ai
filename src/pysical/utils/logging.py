"""프로젝트 공통 로거 설정.

print 대신 로거를 쓰면 시각(timestamp)·모듈명·레벨이 함께 찍혀 디버깅이 쉽다.
어느 모듈에서든 ``get_logger(__name__)`` 로 같은 포맷의 로거를 얻는다.
"""

from __future__ import annotations

import logging

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """모듈용 로거를 돌려준다. 최초 호출 때 한 번만 포맷을 설정한다."""
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        _CONFIGURED = True
    return logging.getLogger(name)
