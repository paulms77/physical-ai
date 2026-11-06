"""YOLOE 검출기 — open-vocabulary(prompt-then-detect)를 BaseDetector로 감싼다.

docs/ARCHITECTURE_v2.md의 실시간 검출 경로다. YOLOE는 대상 어휘를 미리 정하면
텍스트 임베딩을 모델에 구워넣어(set_classes) 순수 CNN 검출기처럼 동작하므로,
GPU 없는 노트북 CPU에서도 실시간에 근접한다.

무거운 ultralytics/torch 의존성이 필요하므로, import는 이 모듈 안에서만 한다.
그래야 기하/파이프라인 코드(순수 NumPy)는 이 의존성 없이도 import·테스트된다.

배포 형태(하이브리드 방침)
------------------------
지금은 검증을 위해 ultralytics로 직접 구동한다(.pt). 검증이 끝나면 A6000에서 ONNX로
export해 노트북은 onnxruntime만으로 돌리는 게 최종 형태다(memory: pysical-3device-arch).
그 교체 지점은 이 클래스의 predict 내부뿐이라, 인터페이스는 그대로 유지된다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io.types import Frame
from ..utils.logging import get_logger
from .types import Detection

log = get_logger(__name__)

# 모델·가중치는 프로젝트 폴더 안(weights/)에만 둔다(사내망 관리 + 디스크 원칙).
WEIGHTS_DIR = Path(__file__).resolve().parents[3] / "weights"


class YoloeDetector:
    """YOLOE(seg) 기반 open-vocabulary 검출기.

    BaseDetector 인터페이스(set_prompt/detect)를 구현한다. MockDetector와 교체
    가능하므로, 파이프라인·시각화·ROS 코드는 이 클래스를 몰라도 된다.

    Parameters
    ----------
    model_name:
        ultralytics YOLOE 가중치 이름. 기본 yoloe-11s-seg(가장 작은 seg).
    conf:
        검출 신뢰도 임계값. **기본 0.4** — open-vocab 검출기는 프롬프트에 있는 물체를
        "억지로라도" 찾으려는 경향(false positive)이 있어, 임계값이 너무 낮으면 없는
        물체를 낮은 신뢰도로 만들어낸다. 0.4면 실제 물체(대개 0.7+)는 통과시키고
        억지 매칭(대개 0.3~0.4)은 거른다. "없으면 없다"를 지키기 위한 기본값.
    device:
        "cpu" 고정(추론 노트북 기준). 개발 중 GPU가 되면 "0"으로.
    """

    def __init__(
        self,
        model_name: str = "yoloe-11s-seg.pt",
        conf: float = 0.4,
        device: str = "cpu",
    ) -> None:
        from ultralytics import YOLOE  # 무거운 import는 여기서만

        WEIGHTS_DIR.mkdir(exist_ok=True)
        model_path = WEIGHTS_DIR / model_name
        # weights/ 에 있으면 그걸, 없으면 이름으로(자동 다운로드) 로드.
        self._model = YOLOE(str(model_path) if model_path.exists() else model_name)
        self._conf = conf
        self._device = device
        self._labels: list[str] = []
        # 라벨 조합별 텍스트 임베딩 캐시. ultralytics는 매 호출마다 MobileCLIP
        # 텍스트 모델을 새로 만들어(~350ms, 큰 임시 할당) 반복 교체가 비싸다.
        # 같은 조합으로 돌아오면 계산을 건너뛴다.
        self._pe_cache: dict[tuple[str, ...], object] = {}
        self._pe_cache_limit = 32
        log.info("YOLOE 로드: %s (device=%s)", model_name, device)

    def set_prompt(self, labels: list[str]) -> None:
        """검출 대상 어휘를 고정한다 (prompt-then-detect의 핵심).

        텍스트 임베딩을 계산해 모델에 심는다. 이 호출 뒤부터 detect는 이 어휘만
        찾는다. 프롬프트를 바꿀 때만 다시 부르면 된다.
        """
        if not labels:
            raise ValueError("빈 프롬프트입니다. 최소 한 개 라벨이 필요합니다.")
        self._labels = list(labels)
        key = tuple(self._labels)
        pe = self._pe_cache.get(key)
        if pe is None:
            pe = self._text_pe(self._labels)
            if len(self._pe_cache) >= self._pe_cache_limit:
                self._pe_cache.clear()  # 상한을 넘으면 통째로 비운다(무한 증가 방지)
            self._pe_cache[key] = pe
        self._model.set_classes(self._labels, pe)
        log.info("프롬프트 설정: %s", self._labels)

    def _text_pe(self, labels: list[str]):
        """라벨의 텍스트 임베딩을 계산한다. **항상 CPU에서 계산한다.**

        MobileCLIP 텍스트 인코더(mobileclip_blt.ts)는 float64 텐서를 포함한
        TorchScript 모듈이라 MPS에 올릴 수 없다::

            TypeError: Cannot convert a MPS Tensor to float64 dtype

        ultralytics는 텍스트 모델을 '지금 모델이 올라가 있는 장치'에 만들려 하므로,
        검출을 MPS로 돌리면 두 번째 set_prompt부터 여기서 터진다(실측: 시작 시
        --prompt는 성공, 이후 음성 교체는 전부 실패). 그래서 임베딩을 만드는 동안만
        모델을 CPU로 내렸다가 되돌린다. 임베딩 계산은 어차피 몇 백 ms짜리 일회성이라
        CPU로도 충분하다.
        """
        import torch

        module = getattr(self._model, "model", None)
        if module is None:
            return self._model.get_text_pe(labels)

        try:
            prev = next(module.parameters()).device
        except StopIteration:
            return self._model.get_text_pe(labels)

        if prev.type == "cpu":
            return self._model.get_text_pe(labels)

        module.to("cpu")
        try:
            pe = self._model.get_text_pe(labels)
            if isinstance(pe, torch.Tensor):
                pe = pe.detach().to("cpu")
        finally:
            module.to(prev)
        return pe

    def detect(self, frame: Frame) -> list[Detection]:
        """현재 프롬프트에 맞는 물체들을 검출한다.

        Returns
        -------
        list[Detection]
            매칭된 모든 인스턴스(README §6: 복수 객체 전부 반환). 각 Detection은
            bbox와 (seg 모델이면) mask를 담는다.
        """
        if frame.rgb is None:
            return []
        if not self._labels:
            raise RuntimeError("set_prompt를 먼저 호출하세요.")

        results = self._model.predict(
            frame.rgb, conf=self._conf, device=self._device, verbose=False
        )
        r = results[0]
        detections: list[Detection] = []

        boxes = r.boxes
        masks = r.masks
        for i in range(len(boxes)):
            cls_idx = int(boxes.cls[i].item())
            label = self._labels[cls_idx] if cls_idx < len(self._labels) else str(cls_idx)
            score = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].cpu().numpy().astype(np.float64)

            mask = None
            if masks is not None:
                # (H, W) bool 마스크로 변환. YOLOE 마스크는 모델 입력 해상도이므로
                # 원본 RGB 크기로 맞춰준다.
                m = masks.data[i].cpu().numpy()
                mask = _resize_mask(m, frame.rgb.shape[:2])

            detections.append(Detection(label=label, score=score, bbox_xyxy=xyxy, mask=mask))

        return detections


def _resize_mask(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """마스크를 목표 (H, W)로 리사이즈해 bool로 만든다."""
    import cv2

    h, w = target_hw
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    return mask > 0.5
