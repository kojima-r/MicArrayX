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


def stft(x, win, step):
    """ STFT (Short Term Fourie Transform)

    一次元短時間フーリエ変換
    Args:
        x (ndarray): 入力信号(一次元信号)
        win (int):   窓関数
        step (int):  シフト幅

    Returns:
        ndarray: STFTの結果：スペクトログラム(M x N)
        - N = 窓幅
        - l = 入力信号の長さ
        - M = [(l - N + step) / step)]

    """

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


def istft(X, win, step):
    """ ISTFT (Short Term Fourie Transform)

    逆一次元短時間フーリエ変換，stft関数の逆変換を行う

    Args:
        X (ndarray): 入力信号(M x N　行列)
        win (int):   窓関数
        step (int):  シフト幅

    Returns:
        ndarray: 逆変換後の信号(l)
        出力信号の長さ: l = [(M - 1) * step + N]

    """


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
    """ 窓関数の適用

    スライディングウィンドウを用いて窓関数を適用

    Args:
        x (ndarray): 入力信号(一次元信号)
        win (int):   窓関数
        step (int):  シフト幅

    Returns:
        ndarray: STFTの結果：スペクトログラム(M x N)
        - N = 窓幅
        - l = 入力信号の長さ
        - M = [(l - N + step) / step)]

    """


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
    """ 多チャンネルwavファイルの読み込み

    Args:
        filename (str): 入力信号(一次元信号)

    Returns:
        Dict[str,Obj]: wavファイルの読み込み
        -  nchannels    : チャンネル数
        -  sample_width : 1サンプルあたりのバイト数, e.g., 2byte(16bit), 3byte(24bit)
        -  framerate    : フレームレート
        -  nframes      : サンプル数
        -  params       : パラメータ一覧をタプル
        -  duration     : 秒
        -  wav          : 波形データnch x nframe
    """


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
    """ 多チャンネルwavファイルの保存

    Args:
        mix_wavdata: 波形データnch x nframe
        output_filename (str): 入力信号(一次元信号)
        sample_width (int): 1サンプルあたりのバイト数, e.g., 2byte(16bit), 3byte(24bit)
        params (List): その他パラメータ一覧をタプル
        framerate (int): フレームレート

    """


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
    """ stft_mch で変換した結果の半分の周波数パワーから全体の周波数パワーを再構成

    Args:
        spec : 周波数スペクトログラム: nch x freqency_bin

    Return:
        周波数スペクトログラム: nch x (freqency_bin x 2)

    """

    spec_c = np.conjugate(spec[:, :0:-1])
    out_spec = np.c_[spec, spec_c[:, 1:]]
    return out_spec


def stft_mch(data, win, step):
    """ 多チャンネルSTFT(stft関数の多チャンネル版)

    一次元短時間フーリエ変換
    Args:
        x (ndarray): 多チャンネル入力信号(Cチャンネルx サンプル数)
        win (int):   窓関数
        step (int):  シフト幅

    Returns:
        ndarray: STFTの結果：スペクトログラム(C x M x N//2+1)
        - N = 窓幅
        - l = 入力信号の長さ
        - M = [(l - N + step) / step)]
    
    Note:
        出力は半分なので注意

    """
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
    """ 多チャンネルISTFT (Short Term Fourie Transform)

    逆一次元短時間フーリエ変換
    stft_mch関数の逆元

    Args:
        X (ndarray): 入力信号(C x M x N//2+1　行列)
        win (int):   窓関数
        step (int):  シフト幅

    Returns:
        ndarray: 逆変換後の信号(C x l)
        - 出力信号の長さ: l = [(M - 1) * step + N]

    """


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


def diff_direction(theta1, theta2):
    """ 角度差分
    Args:
        theta1 (float): theta1
        theta2 (float): theta2

    Returns:
        ndarray: theta1-theta2の角度の差分(-pi < dtheta < pi)

    """
    dtheta = abs(theta1 - theta2)
    if dtheta > 2 * math.pi:
        dtheta -= 2 * math.pi
    if dtheta > math.pi:
        dtheta = 2 * math.pi - dtheta
    return dtheta


def nearest_direction_index(tf_config, theta):
    """ 最も近い伝達関数のインデックスを取得
    
    Args:
        tf_config (Dict): HARK_TF_PARSERで取得できる関数
        theta (float): 角度(rad)

    Returns:
        int: 伝達関数インデックス（このインデクスを使ってインデクスを取得できる）

    """
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
    """ 最も近い伝達関数のインデックスを取得
    
    Args:
        tf_config (Dict): HARK_TF_PARSERで取得できる関数
        target (ndarray): 座標３次元ndarrayで指定

    Returns:
        int: 伝達関数インデックス（このインデクスを使ってインデクスを取得できる）

    """
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
    """ 指定した角度範囲内の伝達関数のインデックスのリストを取得
    
    Args:
        tf_config (Dict): HARK_TF_PARSERで取得できる関数
        theta (float): 角度(rad)
        range_theta (float): 角度(rad)

    Returns:
        List[int]: 伝達関数インデックスのリスト（このインデクスを使ってインデクスを取得できる）

    """
 
    nearest_theta = math.pi
    ret_indeces = []
    for key_index, value in list(tf_config["tf"].items()):
        pos = value["position"]
        th = math.atan2(pos[1], pos[0])  # -pi ~ pi
        dtheta = diff_direction(theta, th)
        if dtheta <= range_theta:
            ret_indeces.append(key_index)
    return ret_indeces

