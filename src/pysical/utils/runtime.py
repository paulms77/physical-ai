"""런타임 환경 제어 — 스레드 수 제한 등.

torch/onnxruntime/OpenBLAS는 기본적으로 가용한 모든 CPU 코어를 쓰려 한다. 공용
머신(부하 높은 A6000 개발 서버)에서는 이게 오히려 경합을 키워 느려진다. 또 실시간
추론 노트북에서도 코어 수에 맞춰 제한하는 게 지연시간 예측에 좋다.

주의: 스레드 관련 환경변수는 **numpy/torch import 전에** 설정해야 효과가 있다.
그래서 이 함수는 스크립트 맨 앞에서 부르는 걸 권장한다.
"""

from __future__ import annotations

import os


def limit_threads(n: int = 4) -> None:
    """CPU 스레드 수를 n으로 제한한다(경합 완화, 지연 예측성↑).

    스크립트 최상단(무거운 import 전)에서 호출하는 게 이상적이다. 이미 torch가
    로드됐다면 torch.set_num_threads도 함께 시도한다.
    """
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(var, str(n))

    # 이미 import된 라이브러리에도 반영 시도(있을 때만).
    try:
        import torch

        torch.set_num_threads(n)
    except ImportError:
        pass
    try:
        import cv2

        cv2.setNumThreads(n)
    except ImportError:
        pass


def pick_device(requested: str = "auto") -> str:
    """검출기를 돌릴 장치를 고른다.

    ``"auto"``면 Apple Silicon의 MPS를 우선한다. 실측(M1, yoloe-11s-seg)에서
    CPU 95ms → MPS 34ms로 약 2.8배 빨랐다. MPS가 없으면 CPU로 떨어진다.

    Parameters
    ----------
    requested:
        ``"auto"`` | ``"cpu"`` | ``"mps"`` | ``"0"``(CUDA) 등. auto가 아니면 그대로 쓴다.

    Returns
    -------
    str
        ultralytics에 넘길 장치 문자열.
    """
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "0"
    except Exception:  # torch가 없거나 조회 실패 — CPU로 간다
        pass
    return "cpu"
