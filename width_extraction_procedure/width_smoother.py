import numpy as np
import abc
import numpy as np
import pandas as pd


try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
except ImportError:
    lowess = None
    print("Warning: can't import statsmodels")

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern
except ImportError:
    GaussianProcessRegressor = None
    print("Warning: can't import sklearn")

def HampelFilter(x0, window_size=7, n=3):
    x0=np.array(x0)
    x=x0.copy()
    m=len(x)
    k = 1.4826  # scaling factor for Gaussian distribution
    for i in range(m):
        start = max(i - window_size//2, 0)
        end = min(i + window_size//2, m - 1)
        median_x=np.median(x0[start:end+1])
        mad=np.median(np.abs(x0[start:end+1]-median_x))
        if np.abs(x0[i]-median_x)>n*k*mad:
            x[i]=median_x
    return x

class BaseSmoother(abc.ABC):
    @abc.abstractmethod
    def smooth(self, t, y):
        pass



class GaussianKernelSmoother(BaseSmoother):
    def __init__(self, h):
        """
        h: 核带宽（时间单位）
        """
        self.h = h

    def smooth(self, t, y):
        t = np.asarray(t)
        y = np.asarray(y)
        n = len(t)
        y_s = np.empty(n, dtype=float)
        for i in range(n):
            dt = t - t[i]
            w = np.exp(-0.5 * (dt / self.h)**2)
            w_sum = w.sum()
            y_s[i] = np.sum(w * y) / w_sum if w_sum != 0 else y[i]
        return y_s


class TimeWeightedMedianSmoother(BaseSmoother):
    def __init__(self, h):
        self.h = h

    @staticmethod
    def weighted_median(values, weights):
        sorter = np.argsort(values)
        v_sorted = values[sorter]
        w_sorted = weights[sorter]
        cumw = np.cumsum(w_sorted)
        cutoff = cumw[-1] / 2.0
        idx = np.searchsorted(cumw, cutoff)
        return v_sorted[min(idx, len(v_sorted)-1)]

    def smooth(self, t, y):
        t = np.asarray(t)
        y = np.asarray(y)
        n = len(t)
        y_s = np.empty(n, dtype=float)
        for i in range(n):
            dt = np.abs(t - t[i])
            w = np.exp(-0.5 * (dt / self.h)**2)
            y_s[i] = self.weighted_median(y, w)
        return y_s


class ContinuousExpSmoother(BaseSmoother):
    def __init__(self, tau=7.0, bidirectional=True):
        self.tau = tau
        self.bidirectional = bidirectional

    def _smooth_oneway(self, t, y):
        n = len(t)
        s = np.empty(n, dtype=float)
        s[0] = y[0]
        for i in range(1, n):
            dt = abs(t[i] - t[i-1])
            alpha = 1 - np.exp(-dt / self.tau)
            s[i] = s[i-1] * (1 - alpha) + alpha * y[i]
        return s

    def smooth(self, t, y):
        t = np.asarray(t)
        y = np.asarray(y)
        s_forward = self._smooth_oneway(t, y)
        if not self.bidirectional:
            return s_forward
        s_backward = self._smooth_oneway(t[::-1], y[::-1])[::-1]
        return 0.5 * (s_forward + s_backward)


# ==========  Gaussian Process  ==========
class GPSmoother(BaseSmoother):
    def __init__(self, kernel="RBF", length_scale=1.0, noise=1e-2):
        if GaussianProcessRegressor is None:
            raise ImportError("scikit-learn not found. Please install scikit-learn to use GPSmoother.")
        self.kernel_type = kernel
        self.length_scale = length_scale
        self.noise = noise
        if kernel == "RBF":
            self.kernel = RBF(length_scale=length_scale)
        elif kernel == "Matern":
            self.kernel = Matern(length_scale=length_scale, nu=1.5)
        else:
            raise ValueError("unknown kernel type: choose 'RBF' or 'Matern'")

    def smooth(self, t, y):
        t = np.asarray(t).reshape(-1, 1)
        y = np.asarray(y)
        gp = GaussianProcessRegressor(kernel=self.kernel, alpha=self.noise**2)
        gp.fit(t, y)
        y_pred, _ = gp.predict(t, return_std=True)
        return y_pred


# ========== 5. LOESS / LOWESS ==========
class LowessSmoother(BaseSmoother):
    def __init__(self, time_window=42.0):
        if lowess is None:
            raise ImportError("scikit-learn not found. Please install scikit-learn to use LowessSmoother.")
        self.time_window = time_window

    def smooth(self, t, y):
        # print(self.time_window/(t[-1]-t[0]))
        result = lowess(y, t, frac=self.time_window/(t[-1]-t[0]), return_sorted=False)
        return result


def GetSmoother(name, **kwargs):
    """
    name: 'gaussian', 'median', 'exp', 'gp', 'lowess'
    """
    mapping = {
        "gaussian": GaussianKernelSmoother,
        "median": TimeWeightedMedianSmoother,
        "exp": ContinuousExpSmoother,
        "gp": GPSmoother,
        "lowess": LowessSmoother
    }
    if name not in mapping:
        raise ValueError(f"unknown smoother name: {name}")
    return mapping[name](**kwargs)
