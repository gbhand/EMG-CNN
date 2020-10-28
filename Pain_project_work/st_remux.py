# Stockwell Remake

import os
import numpy as np

from scipy.signal import iirnotch, butter, lfilter

# Config
# USE_FILTER = 0

""" Returns parameters for three addFilters

Note: iirnotch behaves differently in Python than Matlab
"""
def prepareFilter(fs):
	w0 = 60 / (fs / 2)
	w1 = 120 / (fs / 2)
	w2 = 180 / (fs / 2)
	b1, a1 = butter(8, 15/(fs/2), 'highpass') # highpass filter cutoff at 15Hz
	b2, a2 = iirnotch(w0, 5)  # 60Hz notch
	b3, a3 = iirnotch(w1, 5) # 120Hz notch
	b4, a4 = butter(8, 120/(fs/2), 'lowpass') # 140Hz lowpass
	b5, a5 = iirnotch(w2, 5) # 180Hz notch

	return b1, a1, b2, a2, b3, a3, b4, a4, b5, a5

def addFilter(b, a, data):
	addFiltered_data = lfilter(b, a, data)

	return addFiltered_data

def filter_data(timeseries, fs, win_len, USE_FILTER):
	b1, a1, b2, a2, b3, a3, b4, a4, b5, a5 = prepareFilter(fs)
	
	length = len(timeseries) - len(timeseries) % win_len
	raw = timeseries[0:length]
	
	EMG1 = addFilter(b1, a1, raw);
	EMG2 = addFilter(b2, a2, EMG1);
	EMG3 = addFilter(b3, a3, EMG2);
	EMG4 = addFilter(b4, a4, EMG3);
	EMG5 = addFilter(b5, a5, EMG4);
	EMG53 = addFilter(b3, a3, EMG5);
	EMG52 = addFilter(b2, a2, EMG53);

	if not USE_FILTER:
		EMG52 = raw
	
	return EMG52


def g_window(length, freq, factor):
	vector = np.zeros((2, length))
	vector[0,:] = np.arange(0, length)
	vector[1,:] = np.arange(-length, 0)
	vector = vector**2
	vector = vector * (-factor * 2 * np.pi**2 / freq**2)
	
	return np.sum(np.exp(vector), axis=0)

def st_remux(timeseries):
	minfreq = 5
	maxfreq = 30
	maxfreq = int(np.fix(len(timeseries) / 2))
	# if maxfreq > len(timeseries) / 2:
	# 	maxfreq = int(np.fix(len(timeseries) / 2))
	# 	print(type(maxfreq))
	samplingrate = 4000
	freqsamplingrate = 1
	factor = 1

	# print(type(minfreq))
	# print(type(maxfreq))
	
	n = len(timeseries)
	
	# Compute fft
	vector_fft = np.fft.fft(timeseries)
	tmp = np.roll(vector_fft, -minfreq)
	
	# Preallocate (we like to go FAST)
	pxx = np.zeros(((maxfreq - minfreq + 1), n), dtype='complex128')
	pxx[0,:] = np.fft.ifft(tmp * g_window(n, minfreq, factor))
	
	# Calculate S-transform
	for banana in np.arange(maxfreq - minfreq + 1):
		pxx[banana,:] = np.fft.ifft(np.roll(vector_fft, -(minfreq + banana)) * g_window(n, minfreq + banana, factor))
		
	

	return np.abs(pxx)

def to_array(timeseries, win_len):
	tmp = st_remux(timeseries[0:win_len])
	ydim = tmp.shape[0]
	xdim = tmp.shape[1]
	time_len = len(timeseries)
	zdim = int(time_len / win_len) 
	
	arr = np.zeros((zdim, ydim, xdim))
	
	idx = 0
	for window in np.arange(zdim):
		curr_data = timeseries[idx: idx + win_len]
		pxx = st_remux(curr_data)
		idx = idx + win_len
		arr[window] = pxx
		
	return arr

"""def raw_to_arr(directory, fs, win_len, USE_FILTER): # changed timeseries input to directory
	ydim = 0
	xdim = 0
	

	for index, filename in enumerate(os.listdir(directory)): # get dimensions, make this cleaner
		if filename.endswith('.csv') and index == 1:
			timeseries = np.genfromtxt("" + directory + "/" + filename, delimiter=',')
			filt = filter_data(timeseries, fs, win_len, USE_FILTER)
			arr = to_array(filt, win_len)
			ydim = arr.shape[1]
			xdim = arr.shape[2]
			print('ydim=' + ydim + ' xdim=' + xdim)
		else:
			break
	arr_total = np.zeros((1, 16, 40))

	for filename in os.listdir(directory):
		if filename.endswith('.csv'):
			timeseries = np.genfromtxt("" + directory + "/" + filename, delimiter=',')
			filt = filter_data(timeseries, fs, win_len, USE_FILTER)
			arr = to_array(filt, win_len)
			arr_total = np.concatenate((arr_total, arr))


	return arr_total"""

def raw_to_arr(directory, fs, win_len, USE_FILTER):
    
    ydim = 0
    xdim = 0
    initialpath="/Users/aarushisehgal/Applications/Overall_work/Internship_works/UCSD_project/Pain_project_work/"
    final_path= initialpath + str(directory)
    
    os.chdir(final_path)
    files= os.listdir(final_path)
    print(files)
    
    arr_total = np.zeros((1, 16, 40))
    
    for f in files:
        if f.endswith('.csv'):
            timeseries = np.genfromtxt(f, delimiter=',')
            filt = filter_data(timeseries, fs, win_len, USE_FILTER)
            arr = to_array(filt, win_len)
            arr = np.concatenate((arr_total, arr))
            ydim = arr.shape[1]
            xdim = arr.shape[2]
            print('ydim=' + str(ydim) + ' xdim=' + str(xdim))
            
    return arr

if __name__ == '__main__':
	# high = np.genfromtxt('high.csv', delimiter=',')
	# timeseries = high[0:4000]
	
	# pxx = st_remux(timeseries)
	

	# import matplotlib.pyplot as plt
	# plt.pcolormesh(pxx, cmap='jet')
	# plt.show()


	highdir = 'datasets/high'
	fs = 4000
	win_len = 40
	hi = raw_to_arr(highdir, fs, win_len, 0)