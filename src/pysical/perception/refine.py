"""Focus-and-Refine — 후보 영역만 적응적으로 확대해 재검출한다.

이 프로젝트의 검출 정확도 차별점이다. 흔한 "전체 화면 한 번 검출"의 약점은,
검출 모델이 이미지를 고정 크기(예: 640)로 **축소**해서 보기 때문에 작은/멀리 있는
물체가 몇십 픽셀로 뭉개져 신뢰도가 낮아진다는 것이다.

Focus-and-Refine은 의료 초음파의 "의심 부위 확대 정밀 스캔"과 같은 2단계다::

    1차: 전체 화면 저해상 스캔 → 후보 위치(낮은 신뢰도라도)
    2차: 후보 영역만 잘라 **적응적 배율로 확대** → 재검출 → 신뢰도↑

실측으로 얻은 핵심 교훈(docs/REFINE_FINDINGS 참고)
-------------------------------------------------
- **적당한 확대는 이롭다** — 작은 물체 신뢰도가 2배 이상 오른다(0.22→0.51).
- **과도한 확대는 해롭다** — 4x는 오히려 신뢰도를 떨어뜨린다.
- **선명화/CLAHE는 치명적** — 신뢰도 0으로 만든다. 모델은 사람 눈과 달리 '학습
  데이터의 통계 분포'로 보므로, 인위적 엣지·대비 왜곡은 분포를 벗어나 해롭다.
- 따라서 목표는 '고화질'이 아니라 **'물체를 모델이 학습한 크기대(target_px)로
  되돌리는 것'**. 그래서 배율을 물체 크기에 맞춰 적응적으로 정한다.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..io.types import Frame
from ..utils.logging import get_logger
from .types import Detection

log = get_logger(__name__)

# 모델이 물체를 편하게 보는 크기대(픽셀, 짧은 변 기준). 작은 물체를 이 크기로 확대한다.
# 실측상 이 부근이 신뢰도 최고점. 너무 키우면(>~2.5x 상당) 오히려 나빠진다.
TARGET_OBJECT_PX = 120
MAX_UPSCALE = 3.0


def adaptive_upscale_factor(bbox_xyxy: np.ndarray) -> float:
    """물체가 TARGET_OBJECT_PX가 되도록 확대 배율을 정한다(1.0~MAX_UPSCALE).

    이미 충분히 큰 물체는 확대하지 않는다(1.0). 이게 '무작정 확대'의 실패를 피하는
    핵심 — 모델의 눈높이에 맞추는 것이지 최대한 키우는 게 아니다.
    """
    x1, y1, x2, y2 = bbox_xyxy
    short_side = min(x2 - x1, y2 - y1)
    if short_side <= 0:
        return 1.0
    factor = TARGET_OBJECT_PX / short_side
    return float(np.clip(factor, 1.0, MAX_UPSCALE))


def crop_with_padding(rgb: np.ndarray, bbox_xyxy: np.ndarray, pad_ratio: float = 0.4):
    """bbox를 여백 포함해 잘라내고, 크롭 원점(오프셋)을 함께 돌려준다.

    오프셋은 재검출한 좌표를 다시 전체 이미지 좌표로 되돌릴 때 쓴다.
    """
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    bw, bh = x2 - x1, y2 - y1
    cx1 = int(max(0, x1 - bw * pad_ratio))
    cy1 = int(max(0, y1 - bh * pad_ratio))
    cx2 = int(min(w, x2 + bw * pad_ratio))
    cy2 = int(min(h, y2 + bh * pad_ratio))
    return rgb[cy1:cy2, cx1:cx2], (cx1, cy1)


def refine_detection(
    rgb: np.ndarray,
    detection: Detection,
    detector,
    min_gain: float = 0.0,
):
    """한 검출을 그 영역만 확대 재검출해 더 나은 결과로 갱신한다.

    Parameters
    ----------
    rgb:
        원본 이미지.
    detection:
        1차 검출(대개 신뢰도 낮음).
    detector:
        BaseDetector. 이미 set_prompt로 대상 어휘가 고정돼 있어야 한다.
    min_gain:
        재검출 신뢰도가 이 값 이상 좋아질 때만 교체. 0이면 더 높기만 하면 교체.

    Returns
    -------
    Detection
        갱신된 검출(좌표는 전체 이미지 기준으로 되돌림). 개선이 없으면 원본 그대로.
    """
    factor = adaptive_upscale_factor(detection.bbox_xyxy)
    crop, (ox, oy) = crop_with_padding(rgb, detection.bbox_xyxy)
    if crop.size == 0:
        return detection

    if factor > 1.01:
        # 학습기반 SR(ESPCN)로 확대 — 단순 보간보다 선명하고 검출 신뢰도가 더 오른다
        # (docs/REFINE_FINDINGS.md). 모델이 없으면 Upscaler가 CUBIC으로 폴백한다.
        from .upscale import get_upscaler

        crop_up = get_upscaler(scale=3).upscale(crop, factor=factor)
    else:
        crop_up = crop

    frame = Frame(timestamp=0.0, T_world_camera=np.eye(4), rgb=crop_up)
    redet = detector.detect(frame)
    if not redet:
        return detection

    # 같은 라벨 중 가장 신뢰도 높은 재검출을 고른다.
    same = [d for d in redet if d.label == detection.label]
    best = max(same or redet, key=lambda d: d.score)
    if best.score <= detection.score + min_gain:
        return detection  # 개선 없음 → 원본 유지

    # 재검출 좌표(크롭+확대 기준)를 전체 이미지 좌표로 되돌린다.
    bbox_full = best.bbox_xyxy / factor + np.array([ox, oy, ox, oy])
    mask_full = None
    if best.mask is not None:
        # 확대 크롭의 마스크를 원본 크롭 크기로 줄여 전체 캔버스에 배치.
        mh, mw = crop.shape[:2]
        m = cv2.resize(best.mask.astype(np.uint8), (mw, mh), interpolation=cv2.INTER_NEAREST)
        mask_full = np.zeros(rgb.shape[:2], dtype=bool)
        mask_full[oy : oy + mh, ox : ox + mw] = m > 0

    log.info("refine: %s %.3f → %.3f (배율 %.1fx)",
             detection.label, detection.score, best.score, factor)
    return Detection(
        label=best.label, score=best.score, bbox_xyxy=bbox_full, mask=mask_full
    )


def refine_all(rgb: np.ndarray, detections: list[Detection], detector, conf_below: float = 0.6):
    """신뢰도가 낮은(conf_below 미만) 검출들만 골라 재검출한다.

    이미 확신하는 큰 물체는 건드리지 않고(비용 절약), 애매한 것만 정밀화한다.
    의료 워크플로의 "의심 부위만 확대 정밀 스캔"과 같은 선택적 전략.
    """
    refined = []
    for det in detections:
        if det.score < conf_below:
            refined.append(refine_detection(rgb, det, detector))
        else:
            refined.append(det)
    return refined
