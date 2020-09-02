import pandas as pd
from scipy.signal import iirnotch, butter, lfilter
from numpy import sqrt, mean, abs, dot, cumsum, where, array_split, append
from numpy.fft import fft


# Signal Filtering
""" Returns parameters for three filters

Note: iirnotch behaves differently in Python than Matlab
"""
def prepareFilter(w0, fs):
    b, a = iirnotch(w0, 5)  # 60Hz notch
    d, c = butter(8, 15/(fs/2), 'highpass')
    f, e = butter(8, 120/(fs/2), 'lowpass')

    return b, a, d, c, f, e

def addFilter(b, a, data):
    filtered_data = lfilter(b, a, data)

    return filtered_data



# Feature Generators

""" Root Mean Squared """
def rms(x):
    rms = sqrt(mean(x ** 2))

    return rms

""" Average Rectified Value """
def arv(x):
    arv = mean(abs(x))

    return arv


""" Returns mean power frequency of spectrum

Based on intechopen.com's definition
Should be roughly functionally equivalent to matlab version
"""
def meanfreq(x, win_size):
    sz = int(win_size / 2) + 1
    pxx = abs(fft(x)) ** 2
    ps = pxx[1:sz] * 2e-06
    pwr = sum(ps)
    meanfreq = dot(ps, range(1, sz)) / pwr

    return meanfreq


""" Returns median power frequency of spectrum

Calculates a cumulative sum of the spectrum then locates midpoint
Further improvement may be needed
"""
def medfreq(x, win_size):
    sz = int(win_size / 2) + 1
    pxx = abs(fft(x)) ** 2
    ps = pxx[1:sz] * 2e-06
    cs = cumsum(ps)
    pwr = sum(ps)
    medfreq = where(cs >= pwr * 0.5)[0][0]

    return medfreq



if __name__ == '__main__':
    win_len = 4000   # sampling rate is 4000Hz, so our windows are 1 sec
    fs = 4000
    f0 = 60  #frequency to be filtered out
    w0 = f0/(fs/2)
    b, a, d, c, f, e = prepareFilter(w0, fs)

    file = 'data.csv'
    raw_data = pd.read_csv(file)

    raw_size = len(raw_data)
    num_win = int(raw_size / win_len)

    sub_raw = raw_data[:fs * num_win]  # transforms raw data into 1 sec windows
    sub = array_split(sub_raw, num_win)

    MEF = []
    MDF = []
    ARV = []
    RMS = []

    for m in range(num_win):
        window = sub[m]
        window = addFilter(d, c, window)
        window = addFilter(b, a, window)
        window = addFilter(f, e, window)
        window = addFilter(b, a, window)
        window = window.flatten()

        MEF.append(meanfreq(window, win_len))
        MDF.append(medfreq(window, win_len))
        ARV.append(arv(window))
        RMS.append(rms(window))

    output =  [MEF, MDF, ARV, RMS]