import numpy as np

def _even_degree_list(max_degree: int):
    if max_degree < 0:
        raise ValueError("max_degree must be >= 0")
    return [0] + [k for k in range(2, max_degree+1, 2)]

def _design_even_powers(r_hat: np.ndarray, degs):
    cols = [np.ones_like(r_hat)]
    for d in degs:
        if d == 0:
            continue
        cols.append(r_hat**d)
    return np.column_stack(cols)

def _ridge(A: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    AtA = A.T @ A
    m = AtA.shape[0]
    return np.linalg.solve(AtA + lam*np.eye(m), A.T @ b)

def fit_axis_scalers(ideal_xy_meas: np.ndarray, real_xy_meas: np.ndarray, max_degree: int = 4, lam: float = 1e-5, eps: float = 1e-12):
    X = ideal_xy_meas[:, 0].astype(float)
    Y = ideal_xy_meas[:, 1].astype(float)
    RX = real_xy_meas[:, 0].astype(float)
    RY = real_xy_meas[:, 1].astype(float)

    r = np.sqrt(X*X + Y*Y)
    if len(r) >= 5:
        r_ref = float(np.percentile(r, 95))
    else:
        r_ref = float(np.max(r)) if np.max(r) > 0 else 1.0
    if r_ref <= 0:
        r_ref = 1.0
    r_hat = r / r_ref

    degs = _even_degree_list(max_degree)
    A = _design_even_powers(r_hat, degs)

    mask_x = np.abs(X) > eps
    if not np.any(mask_x):
        raise ValueError("All Ideal_X are ~0; cannot fit Sx.")
    bx = (RX[mask_x] / X[mask_x])
    Ax = A[mask_x, :]
    coef_x = _ridge(Ax, bx, lam)

    mask_y = np.abs(Y) > eps
    if not np.any(mask_y):
        raise ValueError("All Ideal_Y are ~0; cannot fit Sy.")
    by = (RY[mask_y] / Y[mask_y])
    Ay = A[mask_y, :]
    coef_y = _ridge(Ay, by, lam)

    Sx_all = A @ coef_x
    Sy_all = A @ coef_y
    pred_rx = X * Sx_all
    pred_ry = Y * Sy_all

    rmse_x = float(np.sqrt(np.mean((pred_rx[mask_x] - RX[mask_x])**2)))
    rmse_y = float(np.sqrt(np.mean((pred_ry[mask_y] - RY[mask_y])**2)))

    return {
        'deg_list': degs,
        'r_ref': r_ref,
        'coef_x': coef_x,
        'coef_y': coef_y,
        'rmse_x': rmse_x,
        'rmse_y': rmse_y,
    }

def predict_axis_scalers(ideal_xy_all: np.ndarray, params: dict, eps: float = 1e-12):
    X = ideal_xy_all[:, 0].astype(float)
    Y = ideal_xy_all[:, 1].astype(float)
    r = np.sqrt(X*X + Y*Y)
    r_hat = r / max(params['r_ref'], eps)
    A_all = _design_even_powers(r_hat, params['deg_list'])
    Sx = A_all @ params['coef_x']
    Sy = A_all @ params['coef_y']
    pred_x = X * Sx
    pred_y = Y * Sy
    return np.column_stack([pred_x, pred_y])

def classify_axisradial(params: dict):
    degs = params['deg_list']
    coef_x = params['coef_x']
    coef_y = params['coef_y']
    def dS_at_1(coef):
        d = 0.0
        idx = 0
        for k in degs:
            if k == 0:
                idx += 1
                continue
            ak = coef[idx]
            d += k * ak
            idx += 1
        return float(d)
    dx = dS_at_1(coef_x)
    dy = dS_at_1(coef_y)
    def label(val):
        if val > 1e-6:
            return 'Pincushion(+)'
        elif val < -1e-6:
            return 'Barrel(-)'
        else:
            return 'Flat(≈0)'
    return label(dx), label(dy)
