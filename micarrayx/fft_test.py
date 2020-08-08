import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack

pi = np.pi

tdata = np.arange(5999.0) / 300
dt = tdata[1] - tdata[0]

datay = np.sin(pi * tdata) + 2 * np.sin(pi * 2 * tdata)
N = len(datay)
fouriery_1 = fftpack.fft(datay)
print(len(fouriery_1))
print(N)
fouriery_2 = np.fft.fft(datay)
parseval_1 = np.sum(datay ** 2)
parseval_2_1 = np.sum(np.abs(fouriery_1) ** 2) / N
parseval_2_2 = np.sum(np.abs(fouriery_2) ** 2) / N
print(parseval_1)
print(parseval_2_1)
print(parseval_2_2)
print(parseval_1 - parseval_2_1)
print(parseval_1 - parseval_2_2)
