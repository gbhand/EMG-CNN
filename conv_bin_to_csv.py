import os

from numpy import dot, cumsum, where, array_split, savetxt, fromfile, float64, mean, array, sqrt, abs, sum, transpose, reshape, zeros, append
from numpy.fft import fft
from os import listdir, mkdir
from os.path import isfile, join
from scipy.signal import iirnotch, butter, lfilter

import shutil

def prepareFilter(w0, fs):
    b, a = iirnotch(w0, 10)  # 60Hz notch
    d, c = butter(8, 15/(fs/2), 'highpass')
    f, e = butter(8, 120/(fs/2), 'lowpass')

    return b, a, d, c, f, e

def addFilter(b, a, data):
    filtered_data = lfilter(b, a, data)
    return filtered_data


def meanfreq(x, win_size):
    sz = int(win_size / 2) + 1
    pxx = abs(fft(x)) ** 2
    ps = pxx[1:sz] * 2e-06
    pwr = sum(ps)
    meanfreq = dot(ps, range(1, sz)) / pwr

    return meanfreq


def medfreq(x, win_size):
    sz = int(win_size / 2) + 1
    pxx = abs(fft(x)) ** 2
    ps = pxx[1:sz] * 2e-06
    cs = cumsum(ps)
    pwr = sum(ps)
    medfreq = where(cs >= pwr * 0.5)[0][0]

    return medfreq


def rms(x):
    x2 = x*x
    rms = sqrt(sum(x2)/x.size)

    return rms


def arv(x):
    arv = mean(abs(x))

    return arv

#Simple context manager for directory operations
#attrib to Brian J. Hunt https://stackoverflow.com/a/13197763

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


if __name__ == '__main__':
    # sub = [name for name in os.listdir(".") if os.path.isdir(name)]

    initial_path ="/Users/aarushisehgal/Applications/Overall_work/Internship_works/UCSD_project/data_emg/"
    folder= input("Enter the folder name: ")
    dirpath= initial_path+ str(folder)

    fs = 4000 # Sampling rate
    f0 = 60 # Frequency to be removed from signal (Hz)
    w0 = f0/(fs/2)
    b, a, d, c, f, e = prepareFilter(w0, fs)

    win_len = 4000
    max_freq = 500
    rawSize = 4000
    num_win = int(rawSize / win_len)
    print('Number of wins: ', num_win)

    MEF = []
    MDF = []
    ARV = []
    RMS = []
    RAW = []
    raw_out = array([])

    files= os.listdir(dirpath)
    print(files)
    os.chdir(dirpath)

    newfolder= input("Enter destination folder: ")
    pathname2= initial_path + str(newfolder)

    for f in files:

        os.chdir(dirpath)

        filename= f.split('.')
        csvname=filename[0]

        raw = fromfile(f, dtype=float64)
        sub_raw = raw[:fs * num_win]  # transforms raw data into 1 sec windows
        sub = array_split(sub_raw, num_win)
                    
        for m in range(num_win):
            inwin = sub[m]
            dataAF = inwin # filtering is disabled
            # dataAF = addFilter(d, c, dataAF)
            # dataAF = addFilter(b, a, dataAF)
            # dataAF = addFilter(f, e, dataAF)
            # dataAF = addFilter(b, a, dataAF)

            os.chdir(pathname2)

            savetxt(str(csvname)+'.csv', array(dataAF))
            MEF.append(meanfreq(dataAF, win_len))
            MDF.append(medfreq(dataAF, win_len))
            ARV.append(arv(dataAF))
            RMS.append(rms(dataAF))
            # print(reshape(dataAF, (1, rawSize)).shape)
            # RAW.append(transpose(dataAF))
            # RAW.append(reshape(dataAF, (1, rawSize)))
            # RAW.append(dataAF)
            # RAW = RAW.rstrip()
            raw_out = append(raw_out, dataAF)
            # print(raw_out.shape)


        savetxt("ARV1.csv", array(ARV))
        savetxt("RMS1.csv", array(RMS))
        savetxt("MEF1.csv", array(MEF))
        savetxt("MDF1.csv", array(MDF))
        # savetxt("RAW_full.csv", RAW)
        savetxt("RAW.csv", array(raw_out))


    print("Complete, saved CSV files for ", dirpath)