import numpy as np

def fit_affine_2d(xy_src: np.ndarray, xy_dst: np.ndarray):
    if xy_src.shape[0] < 2:
        A = np.eye(2, dtype=float)
        b = np.zeros(2, dtype=float)
        return A, b
    X = np.hstack([xy_src, np.ones((xy_src.shape[0], 1))])
    W, _, _, _ = np.linalg.lstsq(X, xy_dst, rcond=None)
    A = W[:2, :].T
    b = W[2, :]
    return A, b

def rotation_from_affine(A: np.ndarray):
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    theta_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return R, theta_deg
