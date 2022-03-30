# -*- coding: utf-8 -*-
""" command: micarrayx-localize-music
"""
import sys
import numpy as np
import argparse
import json
from scipy import hamming, interpolate, linalg

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import micarrayx
from hark_tf.read_mat import read_hark_tf
from hark_tf.read_param import read_hark_tf_param


def slice_window(x, win_size, step):
    l = x.shape[0]
    N = win_size
    M = int(np.ceil(float(l - N + step) / step))
    out = []
    for m in range(M):
        start = step * m
        if start + N <= l:
            out.append(x[start : start + N])
    o = np.stack(out, axis=0)
    return o


def estimate_spatial_correlation(spec, win_size, step):
    """ 空間相関行列の計算 (einsumを使った実装)

    Args:
        spec (ndarray): 入力信号(channel x #frame x frequency_bin)
        win_size (int):   窓幅
        step (int):  シフト幅

    Returns:
        ndarray: STFTの結果：スペクトログラム(#block x frequency_bin x channel x channel)
        - 複数フレームを用いてブロックを構成する(構成時のパラメータがwin_size, step)

    """
    # ch,frame,spec -> frame,spec,ch
    x = np.transpose(spec, (1, 2, 0))
    # data: frame,block,spec,ch
    data = slice_window(x, win_size, step)
    a = np.transpose(data, (0, 2, 3, 1))
    b = np.transpose(data, (0, 2, 1, 3)).conj()
    c = np.einsum("ijkl,ijlm->ijkm", a, b)
    # out_corr: frame,spec,ch1,ch2
    out_corr = c * 1.0 / win_size
    # print c[0,0]
    return out_corr


def estimate_spatial_correlation2(spec, win_size, step):
    """ 空間相関行列の計算2 (ループを使った実装)

    Args:
        spec (ndarray): 入力信号(channel x #frame x frequency_bin)
        win_size (int):   窓幅
        step (int):  シフト幅

    Returns:
        ndarray: STFTの結果：スペクトログラム(#block x frequency_bin x channel x channel)
        - 複数フレームを用いてブロックを構成する(構成時のパラメータがwin_size, step)

    """

    # ch,frame,spec -> frame,spec,ch
    n_ch = spec.shape[0]
    n_frame = spec.shape[1]
    n_bin = spec.shape[2]
    corr = np.zeros((n_frame, n_bin, n_ch, n_ch), dtype=complex)

    # out_corr: frame,spec,ch1,ch2
    for i in range(n_ch):
        for j in range(n_ch):
            corr[:, :, i, j] = spec[i] * spec[j].conj()
    now_frame = 0
    out = []
    while now_frame + win_size <= n_frame:
        o = np.mean(corr[now_frame : now_frame + win_size], axis=0)
        out.append(o)
        now_frame += step
    return np.array(out)


f = [
    10.0,
    12.5,
    16.0,
    20.0,
    31.5,
    63.0,
    125.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
    4000.0,
    8000.0,
    12500.0,
    16000.0,
    20000.0,
]
w = [
    np.power(10.0, -70.4 / 20.0),
    np.power(10.0, -63.4 / 20.0),
    np.power(10.0, -56.7 / 20.0),
    np.power(10.0, -50.5 / 20.0),
    np.power(10.0, -39.4 / 20.0),
    np.power(10.0, -26.2 / 20.0),
    np.power(10.0, -16.1 / 20.0),
    np.power(10.0, -8.6 / 20.0),
    np.power(10.0, -3.2 / 20.0),
    np.power(10.0, 0.0 / 20.0),
    np.power(10.0, 1.2 / 20.0),
    np.power(10.0, 1.0 / 20.0),
    np.power(10.0, -1.1 / 20.0),
    np.power(10.0, -4.3 / 20.0),
    np.power(10.0, -6.6 / 20.0),
    np.power(10.0, -9.3 / 20.0),
]
f1 = interpolate.interp1d(f, w, kind="cubic")


def A_characteristic(freq):
    """ A特性の計算
    
    予め用意したテーブルからA特性を計算して返す
    
    """
    return f1(freq)


def compute_music_spec(
    spec, src_num, tf_config, df, min_freq_bin=0, win_size=50, step=50, weight_type="uniform"
):
    """ 空間パワー（MUSIC spectrum）の計算

    Args:
        spec (ndarray): 入力信号(channel x #frame x frequency_bin)
        src_num (int): 想定音源数
        tf_config (Dict): HARK_TF_PARSERで取得できる伝達関数
        df: 周波数ビンあたりの周波数： フーリエ変換字のパラメータから計算される
        min_freq_bin (int): 計算に使う最小周波数ビン（ノイズの多い低周波を無視するため）
        win_size (int):   窓幅
        step (int):  シフト幅
        weight_type (str): 周波数ビンの重みのタイプ("uniform": 一様, "A":A特性)

    Returns:
        ndarray: STFTの結果：スペクトログラム(#block x frequency_bin x k)
        - k: 方向のインデックス

    """
    corr = estimate_spatial_correlation(spec, win_size, step)
    power = np.zeros(
        (corr.shape[0], corr.shape[1], len(tf_config["tf"])), dtype=complex
    )
    for frame, freq in np.ndindex((corr.shape[0], corr.shape[1])):
        # normalize correlation
        rxx = corr[frame, freq]
        r = rxx / np.max(np.absolute(rxx)+1.0e-20)
        # eigen value decomposition
        e_val, e_vec = np.linalg.eig(r)
        # sort
        eigen_id = np.argsort(e_val)[::-1]
        e_val = e_val[eigen_id]
        e_vec = e_vec[:, eigen_id]
        e = e_vec[:, src_num:]
        # frequency weight
        if weight_type=="A":
            weight = A_characteristic((min_freq_bin + freq) * df)
        else: # uniform
            weight = 1.0/corr.shape[1]
        # directions
        for k, v in list(tf_config["tf"].items()):
            a_vec = v["mat"][:, min_freq_bin + freq]
            a_vec = a_vec / np.absolute(a_vec)
            s = np.dot(a_vec.conj(), e)
            power[frame, freq, k] = (
                weight * np.dot(a_vec.conj(), a_vec) / np.dot(s, s.conj())
            )
    return power


def save_heatmap_music_spec(outfilename_heat, m_power):
    ax = sns.heatmap(m_power.transpose(), cbar=False, cmap=cm.Greys)
    plt.axis("off")
    sns.despine(
        fig=None,
        ax=None,
        top=False,
        right=False,
        left=False,
        bottom=False,
        offset=None,
        trim=False,
    )
    plt.tight_layout()
    ax.tick_params(labelbottom="off")
    ax.tick_params(labelleft="off")
    plt.savefig(outfilename_heat, bbox_inches="tight", pad_inches=0.0)
    print("[save]", outfilename_heat)


def save_heatmap_music_spec_with_bar(outfilename_heat_bar, m_power):
    plt.clf()
    sns.heatmap(m_power, cbar=True, cmap=cm.Greys)
    plt.savefig(outfilename_heat_bar, bbox_inches="tight", pad_inches=0.0)
    plt.clf()
    print("[save]", outfilename_heat_bar)


def save_spectrogram(outfilename_fft, spec, ch=0):
    x = np.absolute(spec[ch].T)
    ax = sns.heatmap(x[::-1, :], cbar=False, cmap="coolwarm")
    plt.axis("off")
    sns.despine(
        fig=None,
        ax=None,
        top=False,
        right=False,
        left=False,
        bottom=False,
        offset=None,
        trim=False,
    )
    plt.tight_layout()
    ax.tick_params(labelbottom="off")
    ax.tick_params(labelleft="off")
    plt.savefig(outfilename_fft, bbox_inches="tight", pad_inches=0.0)
    print("[save]", outfilename_fft)


def compute_music_power(
    wav_filename,
    tf_config,
    normalize_factor,
    fftLen,
    stft_step,
    min_freq,
    max_freq,
    src_num,
    music_win_size,
    music_step,
    weight_type="uniform",
):
    """ wavファイルからの空間パワー（MUSIC spectrum）の計算

    Args:
        wav_filename (str): 入力wavファイル名
        tf_config (Dict): HARK_TF_PARSERで取得できる伝達関数
        normalize_factor: 入力波形/ normalize_factor　として入力波形の調整をする
        src_num (int): 想定音源数(MUSIC法パラメータ)
        min_freq (int): 計算に使う最小周波数（ノイズの多い低周波を無視するため）
        max_freq (int): 計算に使う最大周波数（ノイズの多い高周波を無視するため）
        fftLen (int):   窓幅
        stft_step (int):  シフト幅
        music_win_size (int):   窓幅
        music_step (int):  シフト幅
        weight_type (str): 周波数ビンの重みのタイプ("uniform": 一様, "A":A特性)

    Returns:
        ndarray: STFTの結果：スペクトログラム(#block x frequency_bin x k)
        - spec: stftの結果
        - m_power: 空間パワー 周波数について平均化した結果 (#block x k)
        - m_full_power: 空間パワー (#block x frequency_bin x k)
        - setting: 各種セッティング

    """
    setting = {}
    # read wav
    print("... reading", wav_filename)
    wav_data = micarrayx.read_mch_wave(wav_filename)
    wav = wav_data["wav"] / normalize_factor
    fs = wav_data["framerate"]
    # print info
    print("# #channels : ", wav_data["nchannels"])
    setting["nchannels"] = wav_data["nchannels"]
    print("# sample size : ", wav.shape[1])
    setting["nsamples"] = wav.shape[1]
    print("# sampling rate : ", fs, "Hz")
    setting["framerate"] = fs
    print("# duration : ", wav_data["duration"], "sec")
    setting["duration"] = wav_data["duration"]

    # reading data
    df = fs * 1.0 / fftLen
    # cutoff bin
    min_freq_bin = int(np.ceil(min_freq / df))
    max_freq_bin = int(np.floor(max_freq / df))
    print("# min freq. :", min_freq_bin * df, "Hz")
    setting["min_freq"] = min_freq_bin * df
    print("# max freq. :", max_freq_bin * df, "Hz")
    setting["max_freq"] = max_freq_bin * df
    print("# freq. step:", df, "Hz")
    setting["freq_step"] = df
    print("# min freq. bin index:", min_freq_bin)
    print("# max freq. bin index:", max_freq_bin)

    # apply STFT
    win = hamming(fftLen)  # ハミング窓
    spec = micarrayx.stft_mch(wav, win, stft_step)
    spec_m = spec[:, :, min_freq_bin:max_freq_bin]
    # apply MUSIC method
    ## power[frame, freq, direction_id]
    print("# src_num:", src_num)
    setting["src_num"] = src_num
    setting["step_ms"] = 1000.0 / fs * stft_step
    setting["music_step_ms"] = 1000.0 / fs * stft_step * music_step
    power = compute_music_spec(
        spec_m,
        src_num,
        tf_config,
        df,
        min_freq_bin,
        win_size=music_win_size,
        step=music_step,
        weight_type=weight_type,
    )
    p = np.sum(np.real(power), axis=1)
    m_power = 10 * np.log10(p + 1.0)
    m_full_power = 10 * np.log10(np.real(power) + 1.0)
    return spec, m_power, m_full_power, setting


def main():
    # argv check
    parser = argparse.ArgumentParser(
        description="applying the MUSIC method to am-ch wave file"
    )
    parser.add_argument(
        "tf_filename",
        metavar="TF_FILE",
        type=str,
        help="HARK2.0 transfer function file (.zip)",
    )
    parser.add_argument(
        "wav_filename", metavar="WAV_FILE", type=str, help="target wav file"
    )
    parser.add_argument(
        "--normalize_factor",
        metavar="V",
        type=int,
        default=32768.0,
        help="normalize factor for the given wave data(default=sugned 16bit)",
    )
    parser.add_argument(
        "--stft_win_size",
        metavar="S",
        type=int,
        default=512,
        help="window sise for STFT",
    )
    parser.add_argument(
        "--stft_step",
        metavar="S",
        type=int,
        default=128,
        help="advance step size for STFT (c.f. overlap=fftLen-step)",
    )
    parser.add_argument(
        "--min_freq",
        metavar="F",
        type=float,
        default=300,
        help="minimum frequency of MUSIC spectrogram (Hz)",
    )
    parser.add_argument(
        "--max_freq",
        metavar="F",
        type=float,
        default=8000,
        help="maximum frequency of MUSIC spectrogram (Hz)",
    )
    parser.add_argument(
        "--music_win_size",
        metavar="S",
        type=int,
        default=50,
        help="block size to compute a correlation matrix for the MUSIC method (frame)",
    )
    parser.add_argument(
        "--music_step",
        metavar="S",
        type=int,
        default=50,
        help="advanced step block size (i.e. frequency of computing MUSIC spectrum) (frame)",
    )
    parser.add_argument(
        "--music_weight_type",
        choices=["uniform","A"],
        default="uniform",
        help="weight of bins of frequencies",
    )
    parser.add_argument(
        "--music_src_num",
        metavar="N",
        type=int,
        default=3,
        help="the number of sound source candidates  (i.e. # of dimensions of the signal subspaces)",
    )
    parser.add_argument(
        "--out_npy",
        metavar="NPY_FILE",
        type=str,
        default=None,
        help="[output] numpy file to save MUSIC spectrogram (time,direction=> power)",
    )
    parser.add_argument(
        "--out_full_npy",
        metavar="NPY_FILE",
        type=str,
        default=None,
        help="[output] numpy file to save MUSIC spectrogram (time,frequency,direction=> power",
    )
    parser.add_argument(
        "--out_fig",
        metavar="FIG_FILE",
        type=str,
        default=None,
        help="[output] fig file to save MUSIC spectrogram (.png)",
    )
    parser.add_argument(
        "--out_fig_with_bar",
        metavar="FIG_FILE",
        type=str,
        default=None,
        help="[output] fig file to save MUSIC spectrogram with color bar(.png)",
    )
    parser.add_argument(
        "--out_spectrogram",
        metavar="FIG_FILE",
        type=str,
        default=None,
        help="[output] fig file to save power spectrogram (first channel) (.png)",
    )
    parser.add_argument(
        "--out_setting",
        metavar="SETTING_FILE",
        type=str,
        default=None,
        help="[output] stting file (.json)",
    )

    args = parser.parse_args()
    if not args:
        quit()

    # read tf
    print("... reading", args.tf_filename)
    tf_config = read_hark_tf(args.tf_filename)

    # print positions of microphones
    # mic_pos=read_hark_tf_param(args.tf_filename)
    # print "# mic positions:",mic_pos

    spec, m_power, m_full_power, setting = compute_music_power(
        args.wav_filename,
        tf_config,
        args.normalize_factor,
        args.stft_win_size,
        args.stft_step,
        args.min_freq,
        args.max_freq,
        args.music_src_num,
        args.music_win_size,
        args.music_step,
    )

    # save setting
    if args.out_setting:
        outfilename = args.out_setting
        fp = open(outfilename, "w")
        json.dump(setting, fp, sort_keys=True, indent=2)
        print("[save]", outfilename)
    # save MUSIC spectrogram
    if args.out_npy:
        outfilename = args.out_npy
        np.save(outfilename, m_power)
        print("[save]", outfilename)
    # save MUSIC spectrogram for each freq.
    if args.out_full_npy:
        outfilename = args.out_full_npy
        np.save(outfilename, m_full_power)
        print("[save]", outfilename)
    # plot heat map
    if args.out_fig:
        save_heatmap_music_spec(args.out_fig, m_power)
    # plot heat map with color bar
    if args.out_fig_with_bar:
        save_heatmap_music_spec_with_bar(args.out_fig_with_bar, m_power)
    # plot spectrogram
    if args.out_spectrogram:
        save_spectrogram(args.out_spectrogram, spec, ch=0)
if __name__ == "__main__":
    main()
