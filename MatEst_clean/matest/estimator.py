import numpy as np
import pandas as pd
from . import models
from . import axisradial
from . import affine

def estimate(df: pd.DataFrame, method: str, degree: int, lam: float, sigma: float):
    mask = df['Real_X'].notna() & df['Real_Y'].notna()
    meas = df[mask].copy()
    stats = {'n_total': len(df), 'n_measured': len(meas), 'method': method,
             'degree_requested': degree, 'lambda': lam, 'sigma': sigma,
             'degree_used': None, 'rmse_x': None, 'rmse_y': None,
             'axis_class': None, 'rotation_deg': None}

    out = df.copy()
    out['Pred_Real_X'] = np.nan
    out['Pred_Real_Y'] = np.nan
    if len(meas) == 0:
        return out, stats

    xy_m = meas[['Ideal_X','Ideal_Y']].to_numpy(float)
    xy_all = df[['Ideal_X','Ideal_Y']].to_numpy(float)
    zx = meas['Real_X'].to_numpy(float)
    zy = meas['Real_Y'].to_numpy(float)

    if method == 'Polynomial':
        d_used = models.auto_degree(degree, len(meas))
        stats['degree_used'] = d_used
        pred_x = models.fit_predict_polynomial_norm(xy_m, zx, xy_all, d_used, lam)
        pred_y = models.fit_predict_polynomial_norm(xy_m, zy, xy_all, d_used, lam)
        pred_mx = models.fit_predict_polynomial_norm(xy_m, zx, xy_m, d_used, lam)
        pred_my = models.fit_predict_polynomial_norm(xy_m, zy, xy_m, d_used, lam)
    elif method == 'Linear':
        d_used = min(1, models.auto_degree(1, len(meas)))
        stats['degree_used'] = d_used
        pred_x = models.fit_predict_polynomial_norm(xy_m, zx, xy_all, d_used, lam)
        pred_y = models.fit_predict_polynomial_norm(xy_m, zy, xy_all, d_used, lam)
        pred_mx = models.fit_predict_polynomial_norm(xy_m, zx, xy_m, d_used, lam)
        pred_my = models.fit_predict_polynomial_norm(xy_m, zy, xy_m, d_used, lam)
    elif method == 'RBF':
        pred_x = models.fit_predict_rbf(xy_m, zx, xy_all, sigma, lam)
        pred_y = models.fit_predict_rbf(xy_m, zy, xy_all, sigma, lam)
        pred_mx = models.fit_predict_rbf(xy_m, zx, xy_m, sigma, lam)
        pred_my = models.fit_predict_rbf(xy_m, zy, xy_m, sigma, lam)
    elif method == 'Radial':
        pred_x, pred_y, pred_mx, pred_my, d_used = models.fit_predict_radial(xy_m, zx, zy, xy_all, degree, lam)
        stats['degree_used'] = d_used
    elif method == 'AxisRadial':
        real_m = np.column_stack([zx, zy])
        A_aff, b_aff = affine.fit_affine_2d(xy_m, real_m)
        R, rot_deg = affine.rotation_from_affine(A_aff)

        real_m_derot = (real_m - b_aff) @ R

        params_ax = axisradial.fit_axis_scalers(xy_m, real_m_derot, max_degree=degree, lam=lam)

        pred_derot_all = axisradial.predict_axis_scalers(xy_all, params_ax)
        pred_derot_meas = axisradial.predict_axis_scalers(xy_m, params_ax)

        pred_all = (pred_derot_all @ R.T) + b_aff
        pred_meas = (pred_derot_meas @ R.T) + b_aff

        pred_x, pred_y   = pred_all[:,0], pred_all[:,1]
        pred_mx, pred_my = pred_meas[:,0], pred_meas[:,1]

        used_degs = [d for d in params_ax['deg_list'] if d != 0]
        stats['degree_used'] = max(used_degs) if used_degs else 0
        stats['axis_class']  = axisradial.classify_axisradial(params_ax)
        stats['rotation_deg'] = float(rot_deg)
    elif method == 'AxisRadial+RBF':
        real_m = np.column_stack([zx, zy])
        A_aff, b_aff = affine.fit_affine_2d(xy_m, real_m)
        R, rot_deg = affine.rotation_from_affine(A_aff)

        real_m_derot = (real_m - b_aff) @ R
        params_ax = axisradial.fit_axis_scalers(xy_m, real_m_derot, max_degree=degree, lam=lam)

        pred_derot_all  = axisradial.predict_axis_scalers(xy_all, params_ax)
        pred_derot_meas = axisradial.predict_axis_scalers(xy_m, params_ax)

        base_all  = (pred_derot_all  @ R.T) + b_aff
        base_meas = (pred_derot_meas @ R.T) + b_aff

        res_x = zx - base_meas[:,0]
        res_y = zy - base_meas[:,1]

        dx_all = models.fit_predict_rbf(xy_m, res_x, xy_all, sigma, lam)
        dy_all = models.fit_predict_rbf(xy_m, res_y, xy_all, sigma, lam)
        dx_m   = models.fit_predict_rbf(xy_m, res_x, xy_m,   sigma, lam)
        dy_m   = models.fit_predict_rbf(xy_m, res_y, xy_m,   sigma, lam)

        pred_x  = base_all[:,0]  + dx_all
        pred_y  = base_all[:,1]  + dy_all
        pred_mx = base_meas[:,0] + dx_m
        pred_my = base_meas[:,1] + dy_m

        stats['rmse_x'] = float(np.sqrt(np.mean((pred_mx - zx)**2)))
        stats['rmse_y'] = float(np.sqrt(np.mean((pred_my - zy)**2)))

        used_degs = [d for d in params_ax['deg_list'] if d != 0]
        stats['degree_used'] = max(used_degs) if used_degs else 0
        stats['axis_class'] = axisradial.classify_axisradial(params_ax)
        stats['rotation_deg'] = float(rot_deg)
    else:
        raise ValueError('Unknown method')

    out['Pred_Real_X'] = pred_x
    out['Pred_Real_Y'] = pred_y
    stats['rmse_x'] = float(np.sqrt(np.mean((pred_mx - zx)**2)))
    stats['rmse_y'] = float(np.sqrt(np.mean((pred_my - zy)**2)))
    return out, stats
