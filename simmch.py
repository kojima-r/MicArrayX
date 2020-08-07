# -*- coding: utf-8 -*-
from scipy import ceil, complex64, float64, hamming, zeros
from scipy.fftpack import fft  # , ifft
from scipy import ifft  # こっちじゃないとエラー出るときあった気がする
from scipy.io.wavfile import read
import wave
import array
import sys
import numpy as np
import numpy.random as npr
import math

# ======
#  STFT
# ======
"""
x : 入力信号
win : 窓関数
step : シフト幅
"""


def stft(x, win, step):
    l = len(x)  # 入力信号の長さ
    N = len(win)  # 窓幅、つまり切り出す幅
    M = int(ceil(float(l - N + step) / step))  # スペクトログラムの時間フレーム数

    new_x = zeros(int(N + ((M - 1) * step)), dtype=float64)
    new_x[:l] = x  # 信号をいい感じの長さにする

    X = zeros([M, N], dtype=complex64)  # スペクトログラムの初期化(複素数型)
    for m in range(M):
        start = int(step * m)
        X[m, :] = fft(new_x[start : start + N] * win)
    return X


# =======
#  iSTFT
# =======
def istft(X, win, step):
    M, N = X.shape
    assert len(win) == N, "FFT length and window length are different."

    l = int((M - 1) * step + N)
    x = zeros(l, dtype=float64)
    wsum = zeros(l, dtype=float64)
    for m in range(M):
        start = int(step * m)
        ### 滑らかな接続
        x[start : start + N] = x[start : start + N] + ifft(X[m, :]).real * win
        wsum[start : start + N] += win
    pos = wsum != 0
    x_pre = x.copy()
    ### 窓分のスケール合わせ
    x[pos] /= wsum[pos]
    return x


def apply_window(x, win, step):
    l = len(x)
    N = len(win)
    M = int(ceil(float(l - N + step) / step))
    step=int(step)

    new_x = zeros(N + ((M - 1) * step), dtype=float64)
    new_x[:l] = x  # zero padding

    X = zeros([M, N], dtype=float64)
    for m in range(M):
        start = step * m
        X[m, :] = new_x[start : start + N] * win
    return X


def read_mch_wave(filename):
    wr = wave.open(filename, "rb")
    # reading data
    data = wr.readframes(wr.getnframes())
    nch = wr.getnchannels()
    wavdata = np.frombuffer(data, dtype="int16")
    fs = wr.getframerate()
    wr.close()
    data = {}
    data["nchannels"] = wr.getnchannels()
    data["sample_width"] = wr.getsampwidth()
    data["framerate"] = wr.getframerate()
    data["nframes"] = wr.getnframes()
    data["params"] = wr.getparams()
    data["duration"] = float(wr.getnframes()) / wr.getframerate()
    mch_wav = []
    for ch in range(nch):
        mch_wav.append(wavdata[ch::nch])
    data["wav"] = np.array(mch_wav)
    return data


def save_mch_wave(
    mix_wavdata, output_filename, sample_width=2, params=None, framerate=16000
):
    a_wavdata = mix_wavdata.transpose()
    out_wavdata = a_wavdata.copy(order="C")
    print("# save data:", output_filename, out_wavdata.shape)
    ww = wave.Wave_write(output_filename)
    if params != None:
        ww.setparams(params)
    else:
        if framerate != None:
            ww.setframerate(framerate)
        if sample_width != None:
            ww.setsampwidth(sample_width)
    if len(out_wavdata.shape) <= 1:
        ww.setnchannels(1)
    else:
        ww.setnchannels(out_wavdata.shape[1])
    ww.setnframes(out_wavdata.shape[0])
    ww.writeframes(array.array("h", out_wavdata.astype("int16").ravel()).tobytes())
    ww.close()


def make_full_spectrogram(spec):
    spec_c = np.conjugate(spec[:, :0:-1])
    out_spec = np.c_[spec, spec_c[:, 1:]]
    return out_spec


def stft_mch(data, win, step):
    fftLen = len(win)
    out_spec = []
    ### STFT
    for m in range(data.shape[0]):
        spectrogram = stft(data[m, :], win, step)
        spec = spectrogram[:, : fftLen // 2 + 1]
        out_spec.append(spec)
    mch_spec = np.stack(out_spec, axis=0)
    return mch_spec


def istft_mch(data, win, step):
    fftLen = len(win)
    out_wav = []
    ### STFT
    for m in range(data.shape[0]):
        spec = data[m, :, :]
        full_spec = make_full_spectrogram(spec)
        resyn_wav = istft(full_spec, win, step)
        out_wav.append(resyn_wav)
    mch_wav = np.stack(out_wav, axis=0)
    return mch_wav


def diff_direction(theta, th):
    dtheta = abs(theta - th)
    if dtheta > 2 * math.pi:
        dtheta -= 2 * math.pi
    if dtheta > math.pi:
        dtheta = 2 * math.pi - dtheta
    return dtheta


def nearest_direction_index(tf_config, theta):
    nearest_theta = math.pi
    nearest_index = None
    for key_index, value in list(tf_config["tf"].items()):
        pos = value["position"]
        th = math.atan2(pos[1], pos[0])  # -pi ~ pi
        dtheta = diff_direction(theta, th)
        if dtheta < nearest_theta:
            nearest_theta = dtheta
            nearest_index = key_index

    return nearest_index


def nearest_position_index(tf_config, target_pos):
    nearest_index = None
    nearest_d = None
    for key_index, value in list(tf_config["tf"].items()):
        pos = value["position"]
        d = np.sum((pos - target_pos) ** 2)
        if nearest_d is None or d < nearest_d:
            nearest_d = d
            nearest_index = key_index
    return nearest_index


def range_direction_index(tf_config, theta, range_theta):
    nearest_theta = math.pi
    ret_indeces = []
    for key_index, value in list(tf_config["tf"].items()):
        pos = value["position"]
        th = math.atan2(pos[1], pos[0])  # -pi ~ pi
        dtheta = diff_direction(theta, th)
        if dtheta <= range_theta:
            ret_indeces.append(key_index)
    return ret_indeces
