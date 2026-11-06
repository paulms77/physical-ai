"""RANSAC 평면 fitting (NumPy 직접 구현).

관측기반 표면추정(docs/ARCHITECTURE_v2.md §2)의 4단계 "Fit surface"다. 물체가 놓인
표면 영역의 3D 점 구름에서 지지 평면 ``ax + by + cz + d = 0`` 을 robust하게 추정한다.

왜 RANSAC인가
-------------
depth 점 구름에는 노이즈와 이상치(물체 자체, 다른 표면)가 섞여 있다. 최소제곱으로
바로 평면을 맞추면 이상치가 결과를 끌고 간다. RANSAC은 매 반복마다 무작위 3점으로
후보 평면을 만들고, 그 평면에 가까운 점(inlier)이 가장 많은 후보를 고른다. 이상치에
강건하다. Open3D의 segment_plane도 같은 알고리즘이며, 여기서는 학습·이식성을 위해
NumPy로 직접 구현한다(무거운 의존성 없음).

이 모듈은 순수 기하다 — 학습도, GPU도, 외부 모델도 필요 없다. CPU에서 사실상 공짜다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Plane:
    """평면 ``n·x + d = 0`` (n은 단위 법선).

    Attributes
    ----------
    normal:
        (3,) 단위 법선 벡터 (a, b, c).
    d:
        평면 방정식의 상수항.
    inlier_ratio:
        fitting에 참여한 점 중 inlier 비율. 평면 품질의 척도(1에 가까울수록 좋음).
    n_inliers:
        inlier 점 개수.
    """

    normal: np.ndarray
    d: float
    inlier_ratio: float = 0.0
    n_inliers: int = 0

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """점(들)에서 평면까지의 부호 있는 거리. n이 단위벡터라 곧 실제 거리다."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return pts @ self.normal + self.d

    def project(self, points: np.ndarray) -> np.ndarray:
        """점(들)을 평면 위로 수직 투영한다.

        접지점을 표면 평면에 투영해 '물체가 표면에 닿는 3D 위치'를 얻을 때 쓴다
        (파이프라인 5단계). 표면 노이즈로 접지점이 평면에서 살짝 떠 있어도 평면에
        붙여준다.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        dist = pts @ self.normal + self.d          # (N,)
        projected = pts - dist[:, None] * self.normal[None, :]
        return projected[0] if np.ndim(points) == 1 else projected


def _plane_from_3_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> tuple[np.ndarray, float] | None:
    """세 점으로 평면(단위 법선, d)을 만든다. 세 점이 일직선이면 None."""
    n = np.cross(p2 - p1, p3 - p1)
    norm = np.linalg.norm(n)
    if norm < 1e-9:                                 # 공선(collinear) → 평면 정의 불가
        return None
    n = n / norm
    d = -float(n @ p1)
    return n, d


def fit_plane_ransac(
    points: np.ndarray,
    distance_threshold: float = 0.01,
    num_iterations: int = 200,
    min_inliers: int = 20,
    seed: int = 0,
) -> Plane | None:
    """3D 점 구름에 RANSAC으로 평면을 fitting한다.

    Parameters
    ----------
    points:
        (N, 3) 3D 점.
    distance_threshold:
        이 거리(미터) 안에 들어오면 inlier로 본다. 기본 1cm.
    num_iterations:
        RANSAC 반복 수. 많을수록 좋은 평면을 찾을 확률↑, 비용↑.
    min_inliers:
        최종 평면이 가져야 할 최소 inlier 수. 못 채우면 None(평면 못 찾음).
    seed:
        난수 시드(재현성). 워크플로/테스트에서 결과 고정.

    Returns
    -------
    Plane | None
        추정된 평면. 유효한 평면을 못 찾으면 None.
    """
    pts = np.asarray(points, dtype=np.float64)
    n_pts = pts.shape[0]
    if n_pts < 3:
        return None

    rng = np.random.default_rng(seed)
    best_inliers: np.ndarray | None = None
    best_model: tuple[np.ndarray, float] | None = None

    for _ in range(num_iterations):
        idx = rng.choice(n_pts, size=3, replace=False)
        model = _plane_from_3_points(pts[idx[0]], pts[idx[1]], pts[idx[2]])
        if model is None:
            continue
        n, d = model
        dist = np.abs(pts @ n + d)
        inliers = dist < distance_threshold
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_model = (n, d)

    if best_model is None or best_inliers is None:
        return None
    if int(best_inliers.sum()) < min_inliers:
        return None

    # 정제(refit): inlier 전체에 최소제곱으로 평면을 다시 맞춰 정확도를 높인다.
    n, d = _refit_plane_lstsq(pts[best_inliers])
    # 정제된 평면 기준으로 inlier를 다시 센다.
    dist = np.abs(pts @ n + d)
    inliers = dist < distance_threshold
    return Plane(
        normal=n,
        d=d,
        inlier_ratio=float(inliers.mean()),
        n_inliers=int(inliers.sum()),
    )


def _refit_plane_lstsq(points: np.ndarray) -> tuple[np.ndarray, float]:
    """inlier 점들에 최소제곱으로 평면을 맞춘다 (PCA: 최소분산 방향이 법선).

    점들의 공분산에서 가장 작은 고유값에 대응하는 고유벡터가 평면의 법선이다.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    # SVD의 마지막 특이벡터 = 최소분산 방향 = 법선.
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    n = vh[-1]
    n = n / np.linalg.norm(n)
    d = -float(n @ centroid)
    return n, d
