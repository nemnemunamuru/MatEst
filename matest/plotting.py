import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm

def _square_padded_limits(xmin, xmax, ymin, ymax, pad_ratio):
    xr = xmax - xmin
    yr = ymax - ymin
    if xr == 0: xr = 1.0
    if yr == 0: yr = 1.0
    xmin -= xr * pad_ratio
    xmax += xr * pad_ratio
    ymin -= yr * pad_ratio
    ymax += yr * pad_ratio
    dx = xmax - xmin
    dy = ymax - ymin
    if dx > dy:
        mid = (ymin + ymax)/2
        ymin = mid - dx/2
        ymax = mid + dx/2
    elif dy > dx:
        mid = (xmin + xmax)/2
        xmin = mid - dy/2
        xmax = mid + dy/2
    return (xmin, xmax), (ymin, ymax)

def _limits_from_arrays(xs, ys, pad_ratio=0.05):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    xs = xs[np.isfinite(xs)]
    ys = ys[np.isfinite(ys)]
    if xs.size == 0 or ys.size == 0:
        return (-1, 1), (-1, 1)
    return _square_padded_limits(np.min(xs), np.max(xs), np.min(ys), np.max(ys), pad_ratio)

def _compute_limits_all(df, pad_ratio: float = 0.05):
    xs = []
    ys = []
    xs.extend(df['Ideal_X'].dropna().tolist())
    ys.extend(df['Ideal_Y'].dropna().tolist())
    m = df['Real_X'].notna() & df['Real_Y'].notna()
    xs.extend(df.loc[m, 'Real_X'].dropna().tolist())
    ys.extend(df.loc[m, 'Real_Y'].dropna().tolist())
    if df['Pred_Real_X'].notna().any():
        xs.extend(df['Pred_Real_X'].dropna().tolist())
        ys.extend(df['Pred_Real_Y'].dropna().tolist())
    return _limits_from_arrays(xs, ys, pad_ratio)

def _compute_limits_realpred_delta(df, pad_ratio: float = 0.05):
    xi = df['Ideal_X'].to_numpy(float)
    yi = df['Ideal_Y'].to_numpy(float)
    rx = df['Real_X'].to_numpy(float)
    ry = df['Real_Y'].to_numpy(float)
    px = df['Pred_Real_X'].to_numpy(float)
    py = df['Pred_Real_Y'].to_numpy(float)
    rdx = rx - xi
    rdy = ry - yi
    pdx = px - xi
    pdy = py - yi
    xs = []
    ys = []
    m = np.isfinite(rdx) & np.isfinite(rdy)
    xs.extend(rdx[m].tolist())
    ys.extend(rdy[m].tolist())
    mp = np.isfinite(pdx) & np.isfinite(pdy)
    xs.extend(pdx[mp].tolist())
    ys.extend(pdy[mp].tolist())
    return _limits_from_arrays(xs, ys, pad_ratio)

def _make_colors(n: int):
    cmap = cm.get_cmap('hsv', max(n, 2))
    return [cmap(i) for i in range(n)]

def make_four_plots(df, stats: dict, dataset_type: str, fig=None):
    if fig is None:
        fig = plt.figure(figsize=(12, 10))
    fig.clf()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    n = len(df)
    colors = _make_colors(n)

    xi = df['Ideal_X'].to_numpy(float)
    yi = df['Ideal_Y'].to_numpy(float)
    rx = df['Real_X'].to_numpy(float)
    ry = df['Real_Y'].to_numpy(float)
    px = df['Pred_Real_X'].to_numpy(float)
    py = df['Pred_Real_Y'].to_numpy(float)
    m_meas = np.isfinite(rx) & np.isfinite(ry)
    m_pred = np.isfinite(px) & np.isfinite(py)

    (xmin_all, xmax_all), (ymin_all, ymax_all) = _compute_limits_all(df)

    # === (1) Ideal ===
    if dataset_type == 'OCT':
        ax1.scatter(df['Ideal_X'], df['Ideal_Y'], s=18, c=colors, alpha=0.9)
    else:  # DIST: gray
        ax1.scatter(df['Ideal_X'], df['Ideal_Y'], s=18, c='gray', alpha=0.9)
    ax1.set_title('Ideal')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal', 'box')
    ax1.set_xlim(xmin_all, xmax_all)
    ax1.set_ylim(ymin_all, ymax_all)
    ax1.grid(True, ls='--', alpha=0.3)

    # === OCT: delta arrays & limits ===
    if dataset_type == 'OCT':
        rdx = rx - xi
        rdy = ry - yi
        pdx = px - xi
        pdy = py - yi
        (xmin_delta, xmax_delta), (ymin_delta, ymax_delta) = _compute_limits_realpred_delta(df)

    # === (2) Input (Real) ===
    if m_meas.any():
        if dataset_type == 'OCT':
            ax2.scatter(rdx[m_meas], rdy[m_meas], s=20,
                        c=[colors[i] for i in np.where(m_meas)[0]], alpha=0.9,
                        label='Real Δ')
            ax2.set_xlim(xmin_delta, xmax_delta)
            ax2.set_ylim(ymin_delta, ymax_delta)
        else:
            ax2.scatter(rx[m_meas], ry[m_meas], s=20, c='tab:blue', alpha=0.9, label='Measured')
            ax2.set_xlim(xmin_all, xmax_all)
            ax2.set_ylim(ymin_all, ymax_all)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No measured points', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Input (Real)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal', 'box')
    ax2.grid(True, ls='--', alpha=0.3)

    # === (3) Predicted (All) ===
    if m_pred.any():
        if dataset_type == 'OCT':
            ax3.scatter(pdx[m_pred], pdy[m_pred], s=18, c=colors, alpha=0.9)
            ax3.set_xlim(xmin_delta, xmax_delta)
            ax3.set_ylim(ymin_delta, ymax_delta)
        else:
            ax3.scatter(px[m_pred], py[m_pred], s=18, c='tab:orange', alpha=0.9)
            dx = (px - xi)
            dy = (py - yi)
            msk = np.isfinite(dx) & np.isfinite(dy)
            ax3.quiver(xi[msk], yi[msk], dx[msk], dy[msk], angles='xy', scale_units='xy', scale=None, color='k', alpha=0.3)
            ax3.set_xlim(xmin_all, xmax_all)
            ax3.set_ylim(ymin_all, ymax_all)
    else:
        ax3.text(0.5, 0.5, 'No predictions yet', ha='center', va='center', transform=ax3.transAxes)

    title3 = 'Predicted (All)'
    if stats.get('method') == 'AxisRadial' and stats.get('axis_class') is not None:
        xlab, ylab = stats['axis_class']
        txt = f"Axis classification\nX: {xlab} Y: {ylab}"
        if stats.get('rotation_deg') is not None:
            txt += f"\nRotation ≈ {stats['rotation_deg']:.3f}°"
        ax3.text(0.02, 0.02, txt, transform=ax3.transAxes, ha='left', va='bottom',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85), fontsize=9)
        title3 += ' [AxisRadial]'
    ax3.set_title(title3)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_aspect('equal', 'box')
    ax3.grid(True, ls='--', alpha=0.3)

    # === (4) Overlay ===
    if dataset_type == 'OCT':
        if m_meas.any():
            ax4.scatter(rdx[m_meas], rdy[m_meas], s=20,
                        c=[colors[i] for i in np.where(m_meas)[0]], alpha=0.8,
                        label='Real Δ (●)')
        if m_pred.any():
            ax4.scatter(pdx[m_pred], pdy[m_pred], s=36, marker='x', c='black', alpha=0.85, label='Pred Δ (×)')
        ax4.set_xlim(xmin_delta, xmax_delta)
        ax4.set_ylim(ymin_delta, ymax_delta)
    else:
        if m_meas.any():
            ax4.scatter(rx[m_meas], ry[m_meas], s=26, c='#0d47a1', alpha=0.85, label='Input (Real)')
        if m_pred.any():
            ax4.scatter(px[m_pred], py[m_pred], s=34, marker='x', c='#1976d2', alpha=0.9, label='Predicted (All)')
        ax4.set_xlim(xmin_all, xmax_all)
        ax4.set_ylim(ymin_all, ymax_all)
    if not (m_meas.any() or m_pred.any()):
        ax4.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Overlay')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_aspect('equal', 'box')
    ax4.grid(True, ls='--', alpha=0.3)
    ax4.legend(loc='best')

    sup = f"N={stats['n_total']} / Measured={stats['n_measured']}  Method={stats['method']}  lambda={stats['lambda']}"
    if stats.get('sigma') and stats['method']=='RBF':
        sup += f"  sigma={stats['sigma']}"
    if stats['degree_used'] is not None:
        sup += f"  degree_used={stats['degree_used']} (req={stats['degree_requested']})"
    if stats['rmse_x'] is not None:
        sup += f"  RMSEx={stats['rmse_x']:.4g}, RMSEy={stats['rmse_y']:.4g}"
    if stats.get('rotation_deg') is not None:
        sup += f"  Rot={stats['rotation_deg']:.3f}°"
    if dataset_type == 'OCT':
        sup += '  View: OCT Δ-mode'

    fig.suptitle(sup, fontsize=12, color='#072b5b')
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    return fig
