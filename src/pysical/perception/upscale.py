"""고화질 확대(super-resolution) — 학습기반 SR로 작은 영역을 선명하게 키운다.

Focus-and-Refine에서 후보 영역을 확대할 때, 단순 보간(cv2.INTER_CUBIC)은 픽셀
사이를 평균으로 메워 뭉개진다(계단현상). 학습기반 SR(ESPCN)은 학습한 디테일을
복원하므로 선명하게 커지고, 검출 신뢰도도 더 오른다.

실측 근거(docs/REFINE_FINDINGS.md §7)
------------------------------------
같은 작은 mouse를 x3 확대했을 때::

    방법        선명도(라플라시안)   검출 신뢰도
    CUBIC             11.4            0.10
    LANCZOS4          11.3            0.00
    ESPCN x3          15.4 (+35%)     0.34 (3.3배)   추론 ~74ms(CPU)

왜 SR은 되고 선명화 필터는 안 되나
--------------------------------
앞서 선명화(sharpen)/CLAHE는 검출을 망쳤다(0.000). 그건 인위적 엣지·대비로 모델의
학습 분포를 벗어나게 만들기 때문이다. 반면 SR은 **학습으로 자연스러운 디테일을 복원**
하므로 분포를 해치지 않는다. "고화질"의 올바른 방법이 SR이다.

모델
----
OpenCV dnn_superres의 ESPCN(x2/x3/x4). GitHub(fannymonori/TF-ESPCN)의 .pb를
weights/에 둔다. 매우 가볍다(~90KB). opencv-contrib-python 필요.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..utils.logging import get_logger

log = get_logger(__name__)

WEIGHTS_DIR = Path(__file__).resolve().parents[3] / "weights"


class Upscaler:
    """ESPCN 기반 super-resolution 확대기(폴백: CUBIC 보간).

    ESPCN은 고정 배율(x2/x3/x4)만 지원하므로, 원하는 배율에 가장 가까운 모델로
    SR한 뒤 필요하면 보간으로 미세 조정한다.

    Parameters
    ----------
    scale:
        SR 배율 (2, 3, 4). 모델이 없으면 자동으로 CUBIC 폴백.
    """

    def __init__(self, scale: int = 3) -> None:
        self._scale = scale
        self._sr = None
        model_path = WEIGHTS_DIR / f"ESPCN_x{scale}.pb"
        try:
            from cv2 import dnn_superres

            if model_path.exists():
                sr = dnn_superres.DnnSuperResImpl_create()
                sr.readModel(str(model_path))
                sr.setModel("espcn", scale)
                self._sr = sr
                log.info("ESPCN x%d 로드: %s", scale, model_path.name)
            else:
                log.warning("ESPCN 모델 없음(%s) → CUBIC 보간으로 폴백", model_path.name)
        except ImportError:
            log.warning("dnn_superres 없음(opencv-contrib 필요) → CUBIC 폴백")

    @property
    def available(self) -> bool:
        """학습기반 SR을 실제로 쓸 수 있는지."""
        return self._sr is not None

    def upscale(self, rgb: np.ndarray, factor: float | None = None) -> np.ndarray:
        """RGB 이미지를 확대한다.

        Parameters
        ----------
        rgb:
            (H, W, 3) uint8.
        factor:
            원하는 총 배율. None이면 모델 기본 배율(self._scale)만큼. 모델 배율과
            다르면 SR 후 보간으로 맞춘다.

        Returns
        -------
        확대된 RGB.
        """
        if self._sr is None:
            # 폴백: CUBIC 보간
            f = factor or self._scale
            return cv2.resize(rgb, (int(rgb.shape[1] * f), int(rgb.shape[0] * f)),
                              interpolation=cv2.INTER_CUBIC)

        # ESPCN은 BGR을 기대(OpenCV 관례).
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out = self._sr.upsample(bgr)  # scale배 확대
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

        # 원하는 배율이 모델 배율과 다르면 보간으로 미세 조정.
        if factor is not None and abs(factor - self._scale) > 0.05:
            target_w = int(rgb.shape[1] * factor)
            target_h = int(rgb.shape[0] * factor)
            out = cv2.resize(out, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        return out


# 프로세스당 하나만 로드해 재사용(모델 로드 비용 절감).
_DEFAULT_UPSCALER: Upscaler | None = None


def get_upscaler(scale: int = 3) -> Upscaler:
    """공용 Upscaler 싱글턴을 돌려준다."""
    global _DEFAULT_UPSCALER
    if _DEFAULT_UPSCALER is None or _DEFAULT_UPSCALER._scale != scale:
        _DEFAULT_UPSCALER = Upscaler(scale)
    return _DEFAULT_UPSCALER
