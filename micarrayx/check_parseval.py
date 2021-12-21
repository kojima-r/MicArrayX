# -*- coding: utf-8 -*-
"""パーセバルの等式をチェックするためのスクリプト（デバッグ用）
"""
from scipy import ceil, complex64, float64, hamming, zeros
from scipy.fftpack import fft  # , ifft
from scipy import ifft  # こっちじゃないとエラー出るときあった気がする
from scipy.io.wavfile import read
import wave
import array
from matplotlib import pylab as pl
import sys
import numpy as np
import numpy.random as npr
import math

import micarrayx
from micarrayx.util import apply_window

from hark_tf.read_mat import read_hark_tf
from hark_tf.read_param import read_hark_tf_param

if __name__ == "__main__":
    # argv check
    if len(sys.argv) < 3:
        print(
            "Usage: check.py <in: src.wav> <in:ch> [<out: log file>]", file=sys.stderr
        )
        quit()
    #
    npr.seed(1234)
    target_ch = int(sys.argv[2])
    wav_filename = sys.argv[1]
    log_file = None
    if len(sys.argv) > 3:
        log_file = sys.argv[3]

    wr = wave.open(wav_filename, "rb")

    # print info
    print("# channel num : ", wr.getnchannels())
    print("# sample size : ", wr.getsampwidth())
    print("# sampling rate : ", wr.getframerate())
    print("# frame num : ", wr.getnframes())
    print("# params : ", wr.getparams())
    print("# sec : ", float(wr.getnframes()) / wr.getframerate())

    # reading data
    data = wr.readframes(wr.getnframes())
    nch = wr.getnchannels()
    wavdata = np.frombuffer(data, dtype="int16")
    fs = wr.getframerate()
    mono_wavdata = wavdata[target_ch::nch]
    wr.close()

    data = mono_wavdata.astype(float64) / 2.0 ** 15
    fftLen = 512
    step = fftLen / 4
    win = hamming(fftLen)
    # win =[1.0]*fftLen
    ### STFT
    spectrogram = micarrayx.stft(data, win, step)
    w_data = apply_window(data, win, step)
    w_sum = np.sum(w_data ** 2, axis=1)

    spec = spectrogram[:, : fftLen // 2 + 1]
    full_spec = micarrayx.make_full_spectrogram(spec)
    s_sum = np.mean((np.abs(spectrogram) ** 2), axis=1)

    print("wav power:", w_sum)
    print("wav #frames:", w_sum.shape)
    print("spec power:", s_sum)
    print("spec #frames:", s_sum.shape)
    rate = s_sum / w_sum
    print("rate:", rate)
    diff = s_sum - w_sum
    print("square error:", np.mean(diff ** 2))

    if log_file != None:
        fp = open(log_file, "w")
        for w, s, r, d in zip(w_sum, s_sum, rate, diff):
            arr = [w, s, r, d]
            fp.write(",".join(map(str, arr)))
