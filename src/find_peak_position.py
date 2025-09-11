from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
import numpy as np
def find_peak_position(noisy_data):
    """
    Find the peak of a noisy dataset using Gaussian KDE and binary search for bandwidth.
    """
    noisy_data=np.array(noisy_data).flatten()
    noisy_data = noisy_data[~np.isnan(noisy_data)]  # Remove NaN values
    if len(noisy_data) < 10:
        return np.nan
    data_min, data_max = np.min(noisy_data), np.max(noisy_data)
    data_range = data_max - data_min
    x = np.linspace(data_min - 0.1*data_range, data_max + 0.1*data_range, 1000)
    bw_l=0.1
    bw_r=0.8
    bw_eps=1e-3

    while(bw_r-bw_l>bw_eps):
        bw_mid=(bw_l+bw_r)/2
        kde_mid = gaussian_kde(noisy_data, bw_method=bw_mid)
        density_mid = kde_mid(x)
        peaks, properties = find_peaks(density_mid)
        _,yy= x[peaks], density_mid[peaks]
        
        if len(peaks)>=2:
            y_sec,y_max = np.partition(yy, -2)[-2:]
            if y_max*0.05<y_sec:
                # double peak
                bw_l=bw_mid
            else:
                bw_r=bw_mid
        else:
            bw_r=bw_mid
    bw_best=(bw_l+bw_r)*0.5+0.1 # smooth the result
    # print(bw_best)
    kde_best = gaussian_kde(noisy_data, bw_method=bw_best)
    density_best = kde_best(x)
    peaks_best, _ = find_peaks(density_best)
    cur_peak= x[peaks_best]
    cur_density= density_best[peaks_best]
    return cur_peak[np.argmax(cur_density)]