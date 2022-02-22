# -*- coding: utf-8 -*-
""" command: micarrayx-sim-tf
"""
from scipy import ceil, complex64, float64, hamming, zeros
from scipy.fftpack import fft
from scipy import ifft
from scipy.io.wavfile import read
import wave
import array
from matplotlib import pylab as pl
import sys
import numpy as np
import numpy.random as npr
import math

import micarrayx
from hark_tf.read_mat import read_hark_tf
from hark_tf.read_param import read_hark_tf_param

# from simuration_random import gen_random_source
# gen_random_source(num,near,far,num_type)
# [{"distance":d,"azimuth":theta,"elevation":0,"type":type_id}]


def make_noise(x):
    rad = npr.rand() * 2 * math.pi
    return math.cos(rad) + 1j * math.sin(rad)


def apply_tf_spec(data, fftLen, step, tf_config, src_index, noise_amp=0):
    """ 入力波形に対して、周波数領域での伝達関数の適用を行い、スペクトログラム形式のまま出力する

    Args:
        data (ndarray): 単チャンネル信号(#sample)
        fftLen (int):   窓関数
        step (int):  シフト幅
        tf_config (Dict): HARK_TF_PARSERで取得できる伝達関数
        src_index (int): 伝達関数インデックス
        noise_amp (float):  ノイズの大きさ

    Returns:
        ndarray: 伝達関数適用後のスペクトログラム(channel x #frame x frequency_bin)

    """

    win = hamming(fftLen)  # ハミング窓
    ### STFT
    spectrogram = micarrayx.stft(data, win, step)
    spec = spectrogram[:, : int(fftLen / 2 + 1)]
    # spec = [4,3,2,1,2*,3*]
    ### Apply TF
    tf = tf_config["tf"][src_index]
    # print "# position:",tf["position"]
    pos = tf["position"]
    th = math.atan2(pos[1], pos[0])  # -pi ~ pi
    # print "# theta(deg):",th/math.pi*180
    out_data = []
    for mic_index in range(tf["mat"].shape[0]):
        tf_mono = tf["mat"][mic_index]
        # print "# src spectrogram:",spec.shape
        # print "# tf spectrogram:",tf_mono.shape
        tf_spec = spec * tf_mono
        spec_c = np.conjugate(tf_spec[:, :0:-1])
        out_spec = np.c_[tf_spec, spec_c[:, 1:]]
        noise_spec = np.zeros_like(out_spec)
        v_make_noise = np.vectorize(make_noise)
        noise_spec = v_make_noise(noise_spec)
        out_spec = out_spec + noise_amp * noise_spec
        out_data.append(out_spec)
    mch_data = np.array(out_data)
    return mch_data


def apply_tf(data, fftLen, step, tf_config, src_index, noise_amp=0):
    """ 入力波形に対して、周波数領域での伝達関数の適用を行い、逆変換を行い、多チャンネル波形の形式で出力する

    Args:
        data (ndarray): 単チャンネル信号(#sample)
        fftLen (int):   窓関数
        step (int):  シフト幅
        tf_config (Dict): HARK_TF_PARSERで取得できる伝達関数
        src_index (int): 伝達関数インデックス
        noise_amp (float):  ノイズの大きさ

    Returns:
        ndarray: 伝達関数適用後の信号(channel x #sample)

    """


    win = hamming(fftLen)  # ハミング窓
    mch_data = apply_tf_spec(data, fftLen, step, tf_config, src_index, noise_amp)
    out_wavdata = []
    for mic_index in range(mch_data.shape[0]):
        spec = mch_data[mic_index]
        ### iSTFT
        resyn_data = micarrayx.istft(spec, win, step)
        out_wavdata.append(resyn_data)
    # concat waves
    mch_wavdata = np.vstack(out_wavdata)
    return mch_wavdata


def main():
    # argv check
    if len(sys.argv) < 5:
        print(
            "Usage: sim_tf.py <in: tf.zip(HARK2 transfer function file)> <in: src.wav> <in:ch> <in:src theta> <in:volume> <out: dest.wav>",
            file=sys.stderr,
        )
        quit()
    #
    npr.seed(1234)
    tf_filename = sys.argv[1]
    wav_filename = sys.argv[2]
    target_ch = int(sys.argv[3])
    src_theta = float(sys.argv[4]) / 180.0 * math.pi
    src_volume = float(sys.argv[5])
    output_filename = sys.argv[6]

    ## read tf
    print("... reading", tf_filename)
    tf_config = read_hark_tf(tf_filename)
    mic_pos = read_hark_tf_param(tf_filename)
    src_index = micarrayx.nearest_direction_index(tf_config, src_theta)
    print("# mic positions  :", mic_pos)
    print("# direction index:", src_index)
    if not src_index in tf_config["tf"]:
        print(
            "Error: tf index", src_index, "does not exist in TF file", file=sys.stderr
        )
        quit()

    ## read wav file
    print("... reading", wav_filename)
    wav_data = micarrayx.read_mch_wave(wav_filename)
    scale = 32767.0
    wav = wav_data["wav"] / scale
    fs = wav_data["framerate"]
    nch = wav_data["nchannels"]
    ## print info
    print("# channel num : ", nch)
    print("# sample size : ", wav.shape)
    print("# sampling rate : ", fs)
    print("# sec : ", wav_data["duration"])
    mono_wavdata = wav[0, :]

    ## apply TF
    fftLen = 512
    step = fftLen / 4
    mch_wavdata = apply_tf(mono_wavdata, fftLen, step, tf_config, src_index)
    a = np.max(mch_wavdata)
    mch_wavdata = mch_wavdata / a
    mch_wavdata = mch_wavdata * scale * src_volume
    ## save data
    micarrayx.save_mch_wave(mch_wavdata, output_filename)

if __name__ == "__main__":
    main()
