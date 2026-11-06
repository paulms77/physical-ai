"""YAML 설정 로딩 유틸.

포트 번호·해상도 같은 값을 코드에 하드코딩하면, 데모 환경이 바뀔 때마다 소스를
고쳐야 한다. 그래서 그런 값은 ``config/*.yaml`` 에 두고 여기서 읽어온다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# 이 파일(src/pysical/utils/config.py)에서 프로젝트 루트까지 3단계 위로 올라간다.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "config"


def load_config(name: str = "phase0") -> dict[str, Any]:
    """``config/<name>.yaml`` 을 읽어 딕셔너리로 돌려준다.

    Parameters
    ----------
    name:
        확장자를 뺀 설정 파일 이름. 기본값은 Phase 0 설정.
    """
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"설정 파일이 없습니다: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
