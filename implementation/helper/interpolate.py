import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
import numpy.polynomial.polynomial as poly

# Different methods for interpolating / imputing missing data

def interpolate_missing(x, y, z, method, take=1, use=1):
    missing_indices = np.where(np.isnan(x))[0]
    x = np.array([elem if i % use == 0 else np.nan for (i, elem) in enumerate(x)])[::take]
    y = np.array([elem if i % use == 0 else np.nan for (i, elem) in enumerate(y)])[::take]
    z = np.array([elem if i % use == 0 else np.nan for (i, elem) in enumerate(z)])[::take]

    match method:
        case 'none':
            pass
        case 'linear':
            x, y, z = interpolate_nan_linear(x, y, z)
        case 'polynomial':
            x, y, z = interpolate_nan_polynomial(x, y, z)
        case 'gpr':
            x, y, z = interpolate_nan_gpr(x, y, z)
    return x, y, z, missing_indices


def interpolate_nan_linear(x, y, z):
    # interpolate x, y, and z coordinates
    nonz_idcs = lambda l: l.nonzero()[0]
    x[np.isnan(x)] = np.interp(nonz_idcs(np.isnan(x)), nonz_idcs(~np.isnan(x)), x[~np.isnan(x)])
    y[np.isnan(y)] = np.interp(nonz_idcs(np.isnan(y)), nonz_idcs(~np.isnan(y)), y[~np.isnan(y)])
    z[np.isnan(z)] = np.interp(nonz_idcs(np.isnan(z)), nonz_idcs(~np.isnan(z)), z[~np.isnan(z)])
    return x, y, z


def interpolate_nan_polynomial(x, y, z, deg=80):
    nonz_idcs = lambda l: l.nonzero()[0]
    bad_indices = nonz_idcs(np.isnan(x))
    good_indices = nonz_idcs(~np.isnan(x))

    # fit missing points to polynomial
    coefs_x = poly.polyfit(good_indices, x[good_indices], deg)
    x[bad_indices] = poly.polyval(bad_indices, coefs_x)
    coefs_y = poly.polyfit(good_indices, y[good_indices], deg)
    y[bad_indices] = poly.polyval(bad_indices, coefs_y)
    coefs_z = poly.polyfit(good_indices, z[good_indices], deg)
    z[bad_indices] = poly.polyval(bad_indices, coefs_z)
    return x, y, z


kernel = ConstantKernel(1000.0, (1e-3, 1e3)) * Matern(0.01, (1e-3, 1e3), 1.5)
def interpolate_nan_gpr(x, y, z):
    """Interpolate using sklearn Gaussian Process Regressor"""
    good_indices = np.nonzero(~np.isnan(y))[0]
    bad_indices = np.nonzero(np.isnan(y))[0]
    if len(bad_indices) == 0 or len(good_indices) == 0: return x, y, z

    X = good_indices.reshape(-1, 1)

    gpr = GaussianProcessRegressor(kernel, n_restarts_optimizer=10, alpha=1e-10, normalize_y=True)

    x[bad_indices] = gpr \
        .fit(X, x[good_indices].reshape(-1, 1)) \
        .predict(bad_indices.reshape(-1, 1)).T

    y[bad_indices] = gpr \
        .fit(X, y[good_indices].reshape(-1, 1)) \
        .predict(bad_indices.reshape(-1, 1)).T

    z[bad_indices] = gpr \
        .fit(X, z[good_indices].reshape(-1, 1)) \
        .predict(bad_indices.reshape(-1, 1)).T

    return x, y, z


def interpolate_nan_gpr_uncertainty(x, y, z):
    good_indices = np.nonzero(~np.isnan(x))[0]
    bad_indices = np.nonzero(np.isnan(x))[0]

    X = good_indices.reshape(-1, 1)
    gpr = GaussianProcessRegressor(kernel, n_restarts_optimizer=10, alpha=1e-10, normalize_y=True)

    x_pred, x_std = gpr \
        .fit(X, x[good_indices].reshape(-1, 1)) \
        .predict(np.array(range(len(x))).reshape(-1, 1), return_std=True)

    x = x_pred.T

    y_pred, y_std = gpr \
        .fit(X, y[good_indices].reshape(-1, 1)) \
        .predict(np.array(range(len(x))).reshape(-1, 1), return_std=True)

    y = y_pred.T

    z_pred, z_std = gpr \
        .fit(X, z[good_indices].reshape(-1, 1)) \
        .predict(np.array(range(len(x))).reshape(-1, 1), return_std=True)

    z = z_pred.T

    return x, y, z, x_std, y_std, z_std
