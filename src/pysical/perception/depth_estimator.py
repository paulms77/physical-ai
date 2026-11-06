"""Monocular metric depth 추정 — 방법 B (docs/ARCHITECTURE_v2.md §1-B).

RGB 한 장에서 depth를 추정해 frame.depth_mono 를 채운다. LiDAR가 없는 기기에서도
3D 위치추정이 가능하게 하는 경로다.

모델: Depth Anything V2 Small (indoor), **ONNX Runtime(CPU)** 로 구동.
- 우리 아키텍처 원칙(추론 노트북은 onnxruntime)에 정확히 맞는다.
- 사내망에서 HF CDN이 막혀, GitHub releases(fabio-sim/Depth-Anything-ONNX)의 ONNX를 쓴다.
- transformers/torch 없이 onnxruntime + numpy 만으로 돈다(가볍다).

metric 스케일 주의
-----------------
Depth Anything V2의 출력은 기본적으로 **상대(relative) depth**다. 진짜 미터가 되려면
LiDAR/ARKit로 스케일을 보정해야 한다(README §11의 리스크). 이 추정기는 raw 출력을
그대로 주고, 보정은 :func:`calibrate_scale` 로 호출자가 선택한다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..utils.logging import get_logger

log = get_logger(__name__)

# 모델은 프로젝트 폴더 안(weights/)에만 둔다(사내망 관리 + 디스크 원칙).
WEIGHTS_DIR = Path(__file__).resolve().parents[3] / "weights"
DEFAULT_MODEL = WEIGHTS_DIR / "depth_anything_v2_vits_indoor.onnx"

# ImageNet 정규화 상수 (Depth Anything V2 전처리 표준).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class MonocularDepthEstimator:
    """Depth Anything V2(ONNX/CPU) monocular depth 추정기.

    Parameters
    ----------
    model_path:
        ONNX 모델 경로. 기본은 weights/의 small indoor 모델.
    input_size:
        추론 입력 정사각 크기(14의 배수 권장). 작을수록 빠르고 덜 정밀.
        518이 원 논문 기본, CPU에서는 364~518 사이가 현실적.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        input_size: int = 364,
    ) -> None:
        import onnxruntime as ort

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"depth 모델이 없습니다: {model_path}\n"
                "GitHub(fabio-sim/Depth-Anything-ONNX) releases에서 받아 weights/에 두세요."
            )
        self._sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._sess.get_inputs()[0].name
        # 14의 배수로 맞춘다(모델 요구).
        self._size = (input_size // 14) * 14
        log.info("Depth Anything V2(ONNX) 로드: %s (input=%d)", model_path.name, self._size)

    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        """RGB (H, W, 3) uint8 → depth (H, W) float32 (원본 해상도, 상대 depth).

        절차: 리사이즈 → ImageNet 정규화 → (1,3,H,W) → onnx 추론 → 원본 크기 복원.
        """
        h, w = rgb.shape[:2]
        inp = self._preprocess(rgb)
        depth = self._sess.run(None, {self._input_name: inp})[0]  # (1, H', W')
        depth = np.asarray(depth).squeeze()
        # 원본 RGB 해상도로 복원(위치추정이 RGB 픽셀과 일치해야 하므로).
        if depth.shape != (h, w):
            depth = cv2.resize(depth.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        return depth.astype(np.float32)

    def _preprocess(self, rgb: np.ndarray) -> np.ndarray:
        """RGB uint8 → (1, 3, size, size) float32, ImageNet 정규화."""
        img = cv2.resize(rgb, (self._size, self._size), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        img = np.transpose(img, (2, 0, 1))          # HWC → CHW
        return img[None, :, :, :].astype(np.float32)  # 배치 차원 추가


def calibrate_scale(
    depth_mono: np.ndarray,
    depth_ref: np.ndarray,
    ref_valid: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """상대 depth를 참조 depth(LiDAR/ARKit)로 metric 스케일 보정한다.

    두 depth가 겹치는 유효 픽셀에서 최소제곱 스케일 s를 구해 depth_mono * s 가
    참조에 맞도록 한다. "LiDAR 없이 가능한가"의 순수성을 위해, 데모에서는 보정 없이
    먼저 재고 필요할 때만 이걸 쓴다(README §11 방침).

    참고: Depth Anything은 값이 '가까울수록 큼'(inverse depth 경향)일 수 있어, 실제
    보정 시에는 부호/역수 관계도 함께 맞춰야 한다. 여기서는 단순 스케일만 제공하며,
    Phase 3에서 LiDAR 대비 관계를 실측해 확정한다.

    Returns
    -------
    (보정된 depth, 스케일 s)
    """
    m = np.isfinite(depth_mono) & np.isfinite(depth_ref) & (depth_mono > 0) & (depth_ref > 0)
    if ref_valid is not None:
        m &= ref_valid.astype(bool)
    if m.sum() < 10:
        return depth_mono, 1.0
    a = depth_mono[m]
    b = depth_ref[m]
    s = float(np.dot(a, b) / np.dot(a, a))
    return depth_mono * s, s
