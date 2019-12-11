# -*- coding: utf-8 -*-
"""
Generate frequency-domain features of collected electromagnetic data.
First perform three test cases to verify expected/proper results.
Then compute spectrum. 
@author: Sg
"""
import numpy as np
import pandas as pd
import scipy.fftpack
from scipy import signal
from scipy.signal import hann
import matplotlib.pyplot as plt
from IPython import get_ipython
from sklearn.preprocessing import Imputer
get_ipython().run_line_magic('matplotlib', 'qt')


class Spectrum:
    def __init__(self, y, N, T, pad, pwr, scl):
        w = hann(N)
        ywf = scipy.fftpack.fft(y*w)
        xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
        plt.figure()
        plt.plot(y[:y.size//scl - pad])
        fig, ax = plt.subplots()
        ax.plot(xf, (2.0/N * np.abs(ywf[0:N//2])**pwr))
        plt.show()
        
class Preprocess:
    def __init__(self, data):
        imputer = Imputer(missing_values='NaN', strategy='mean')
        imputer.fit(data.values.reshape(-1, 1))  # fit single column
        self.data = imputer.transform(data.values.reshape(-1, 1))


# Initialize test case parameters
N = 2400  # Numb signal samplepoints (make same as numb points radar data)
T = 1.0 / 800.0  # sample spacing (reciprocal of test sample rate)
x = np.linspace(0.0, N*T, N)

# Test case 1: Pure DC signal
y = np.ones(x.size)  # create signals at 0 Hz
spectrum = Spectrum(y, N, T, 0, 2, 4)

# Test case 2: Two pure tones at 50 Hz and 80 Hz
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
spectrum = Spectrum(y, N, T, 0, 2, 4)

# Test case 3: Tone at 50 Hz with additive noise
y = np.sin(50.0 * 2.0 * np.pi * x) + 0.5 * np.random.randn(x.size)
spectrum = Spectrum(y, N, T, 0, 1, 4)

# Compute spectrum of electromagnetic data
filename = 'EM_Data.txt'
df = pd.read_csv(filename)

# Preprocessing
data = df.iloc[:, 1]
preprocess = Preprocess(data)
y = preprocess.data

# Segment the data
segment = slice(1800, 1950)
y = signal.detrend(y[segment].flatten(), type='constant')
pad = 1000  # Pad data
y = np.pad(y, (0, pad), mode='constant')  # pad data

# Initialize parameters
T = 1.0/10.0
N = y.size
spectrum = Spectrum(y, N, T, pad, 2, 1)
