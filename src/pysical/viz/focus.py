"""Auto-Focus — 지정 대상을 추출·중앙 정렬·확대하고 계측 정보를 얹는다.

이 프로젝트의 차별점이다. 흔한 "물체 잡아 3D 좌표 내기"를 넘어, **의료 워크플로의
기능적 버전**을 구현한다:

    의료: "병변 보여줘" → 병변 세그멘테이션 → 정중앙 확대 → 크기/깊이 계측
    우리: "사과 보여줘" → 객체 세그멘테이션 → 정중앙 확대 → 3D 크기/거리 계측

연구진/의료진이 대상을 더 세밀하게 볼 수 있도록, 지정 대상만 뽑아 화면 중앙에
크게 정렬해 보여준다. 지금은 병변 대신 일반 객체로 '기능'을 증명한다.

이 모듈은 순수 이미지/기하 연산(cv2+numpy)이라 CPU에서 가볍고, 검출·depth 결과가
있으면 바로 얹힌다.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..io.types import CameraIntrinsics
from ..perception.types import Detection, ObjectLocation


def crop_object(
    rgb: np.ndarray,
    detection: Detection,
    pad_ratio: float = 0.25,
    use_mask: bool = True,
) -> np.ndarray:
    """대상 물체 영역을 잘라낸다(주변 여백 pad_ratio 포함).

    마스크가 있으면 배경을 어둡게 눌러 대상을 도드라지게 한다(의료 뷰어에서 병변만
    강조하는 것과 같은 효과).
    """
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = detection.bbox_xyxy
    bw, bh = x2 - x1, y2 - y1
    # 여백을 붙여 문맥을 조금 남긴다.
    cx1 = int(max(0, x1 - bw * pad_ratio))
    cy1 = int(max(0, y1 - bh * pad_ratio))
    cx2 = int(min(w, x2 + bw * pad_ratio))
    cy2 = int(min(h, y2 + bh * pad_ratio))

    view = rgb.copy()
    if use_mask and detection.mask is not None:
        # 마스크 밖을 어둡게(40%) — 대상만 밝게 남긴다.
        dark = (view * 0.4).astype(np.uint8)
        m = detection.mask[..., None]
        view = np.where(m, view, dark)

    return view[cy1:cy2, cx1:cx2]


def make_focus_view(
    rgb: np.ndarray,
    detection: Detection,
    out_size: int = 512,
    location: ObjectLocation | None = None,
    size_m: tuple[float, float] | None = None,
) -> np.ndarray:
    """대상을 중앙에 확대 정렬한 정사각 포커스 뷰를 만든다.

    Parameters
    ----------
    rgb:
        원본 이미지.
    detection:
        포커스할 대상.
    out_size:
        출력 정사각 한 변(px).
    location:
        3D 위치추정 결과(있으면 거리/좌표를 오버레이).
    size_m:
        (가로, 세로) 실측 크기(미터). 있으면 계측 표시.

    Returns
    -------
    (out_size, out_size, 3) 확대·정렬·계측된 뷰.
    """
    crop = crop_object(rgb, detection)
    ch, cw = crop.shape[:2]
    if ch == 0 or cw == 0:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)

    # 종횡비를 유지하며 out_size 안에 맞춰 확대(letterbox로 중앙 정렬).
    scale = min(out_size / cw, out_size / ch)
    nw, nh = int(cw * scale), int(ch * scale)
    # 확대(크게 키울 때)면 학습기반 SR(ESPCN)로 선명하게. 축소면 보간.
    if scale > 1.2:
        from ..perception.upscale import get_upscaler

        resized = get_upscaler(scale=3).upscale(crop, factor=scale)
        # SR 후 정확한 목표 크기로 맞춤(letterbox 정렬용).
        resized = cv2.resize(resized, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC)

    canvas = np.zeros((out_size, out_size, 3), dtype=np.uint8)
    ox, oy = (out_size - nw) // 2, (out_size - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized

    _overlay_measurements(canvas, detection, location, size_m)
    return canvas


def _overlay_measurements(
    canvas: np.ndarray,
    detection: Detection,
    location: ObjectLocation | None,
    size_m: tuple[float, float] | None,
) -> None:
    """포커스 뷰 위에 라벨·거리·크기 계측 정보를 얹는다."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [f"{detection.label}  conf={detection.score:.2f}"]

    # OpenCV 기본 폰트는 한글을 못 그리므로 계측 라벨은 영문으로 둔다(깨짐 방지).
    if location is not None:
        p = location.position_world
        dist = float(np.linalg.norm(p))
        lines.append(f"dist {dist:.2f} m")
        lines.append(f"world ({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})m [{location.method}]")

    if size_m is not None:
        lines.append(f"size {size_m[0]*100:.1f} x {size_m[1]*100:.1f} cm")

    # 상단에 반투명 정보바.
    bar_h = 22 * len(lines) + 10
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    for i, text in enumerate(lines):
        cv2.putText(canvas, text, (10, 22 * (i + 1)), font, 0.55, (0, 255, 128), 1, cv2.LINE_AA)

    # 중앙 십자선 — 대상이 정중앙에 정렬됐음을 강조(의료 뷰어 관례).
    h, w = canvas.shape[:2]
    cv2.drawMarker(canvas, (w // 2, h // 2), (0, 200, 255),
                   cv2.MARKER_CROSS, markerSize=30, thickness=1)


def estimate_object_size(
    detection: Detection,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[float, float] | None:
    """대상의 실측 크기(가로, 세로, 미터)를 추정한다.

    bbox의 픽셀 크기와 대상까지의 depth로 실제 크기를 역산한다(핀홀 모델):

        실제크기 = 픽셀크기 / 초점거리 * 거리

    의료 버전에서 "병변 크기 몇 mm"에 해당하는 계측이다.
    """
    x1, y1, x2, y2 = detection.bbox_xyxy
    # 대상 중심 부근의 depth 중앙값
    cu, cv_ = int((x1 + x2) / 2), int((y1 + y2) / 2)
    win = depth[max(0, cv_ - 3): cv_ + 4, max(0, cu - 3): cu + 4]
    valid = win[np.isfinite(win) & (win > 0)]
    if valid.size == 0:
        return None
    z = float(np.median(valid))

    width_m = (x2 - x1) / intrinsics.fx * z
    height_m = (y2 - y1) / intrinsics.fy * z
    return (width_m, height_m)
