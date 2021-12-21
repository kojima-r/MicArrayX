# -*- coding: utf-8 -*-
import sys
import numpy as np
import numpy.random as npr
from scipy import hamming, interpolate
import scipy

import matplotlib

matplotlib.use("Agg")
import sys
import numpy as np

np.random.seed(0)
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json

import micarrayx
from hark_tf import read_hark_tf
from hark_tf.read_param import read_hark_tf_param
from hark_tf.sorting_mat import permutation_hark_tf

from optparse import OptionParser

def main():
    usage = "usage: %s tf [options] <in: src.wav> <out: dest.wav>" % sys.argv[0]
    parser = OptionParser()

    parser.add_option(
        "--min-freq",
        dest="min_freq",
        help="minimum frequency (Hz)",
        default=2000,
        type=float,
        metavar="F",
    )

    parser.add_option(
        "--max-freq",
        dest="max_freq",
        help="maximum frequency (Hz)",
        default=8000,
        type=float,
        metavar="F",
    )

    parser.add_option(
        "--stft-win",
        dest="stft_win",
        help="window size for STFT",
        default=512,
        type=int,
        metavar="W",
    )

    parser.add_option(
        "--stft-adv",
        dest="stft_adv",
        help="advance step size for STFT",
        default=160,
        type=int,
        metavar="W",
    )
    parser.add_option(
        "--out-npy",
        dest="npy_file",
        help="[output] numpy MUSIC spectrogram file (dB :same with HARK)",
        default=None,
        type=str,
        metavar="FILE",
    )

    parser.add_option(
        "--out-full-npy",
        dest="npy_full_file",
        help="[output] numpy MUSIC spectrogram file for each frequency bin (raw spectrogram)",
        default=None,
        type=str,
        metavar="FILE",
    )

    parser.add_option(
        "--out-loc",
        dest="loc_file",
        help="[output] localization file(.json)",
        default=None,
        type=str,
        metavar="FILE",
    )

    parser.add_option(
        "--out-2d",
        dest="loc2d_file",
        help="[output] 2d (.json)",
        default=None,
        type=str,
        metavar="FILE",
    )

    parser.add_option(
        "--plot-h",
        dest="plot_h_file",
        help="[output] heatmap file",
        default=None,
        type=str,
        metavar="FILE",
    )

    parser.add_option(
        "--plot-hb",
        dest="plot_hb_file",
        help="[output] heatmap file with bar",
        default=None,
        type=str,
        metavar="FILE",
    )

    parser.add_option(
        "--plot-fft",
        dest="plot_fft_file",
        help="[output] spectrogram",
        default=None,
        type=str,
        metavar="FILE",
    )

    (options, args) = parser.parse_args()

    # argv check
    if len(args) < 2:
        print(
            "Usage: music.py <in: tf.zip(HARK2 transfer function file)> <in: src.wav>",
            file=sys.stderr,
        )
        quit()
    #
    # read tf
    npr.seed(1234)
    tf_filename = args[0]
    tf_config = read_hark_tf(tf_filename)
    mic_pos = read_hark_tf_param(tf_filename)
    permutation = permutation_hark_tf(tf_filename)

    print("# mic positions:", mic_pos)
    # read wav
    music_filename = args[1]
    print("... reading", music_filename)
    power = np.load(music_filename)
    nch = len(mic_pos)
    # print info
    print("# channel num : ", nch)

    # reading data
    fftLen = int(tf_config["nfft"])
    fs = float(tf_config["samplingRate"])
    df = fs * 1.0 / fftLen
    step = options.stft_adv  # fftLen / 4
    step_ms = fs / step
    # cutoff bin
    min_freq = options.min_freq
    max_freq = options.max_freq
    min_freq_bin = int(np.ceil(min_freq / df))
    max_freq_bin = int(np.floor(max_freq / df))
    print("# min freq:", min_freq)
    print("# max freq:", max_freq)
    print("# min fft bin:", min_freq_bin)
    print("# max fft bin:", max_freq_bin)

    # power: frame, freq, direction_id
    # music_win=options.music_win
    # music_step=options.music_adv
    # music_step_ms=music_step*step_ms
    #
    epsilon = 0.01
    print(power.shape)
    count = 0
    pos_list = []
    for key, val in list(permutation.items()):
        if np.abs(tf_config["positions"][val][3] - 0) < epsilon:
            pos_list.append(tf_config["positions"][val])
            count += 1
    arr = power.shape
    arr2 = (arr[0], arr[1], count)
    out_power = np.zeros(arr2)
    count = 0
    for key, val in list(permutation.items()):
        if np.abs(tf_config["positions"][key][3] - 0) < epsilon:
            # print key
            out_power[:, :, count] = np.absolute(power[:, :, key])
            count += 1
    p = np.sum(np.real(out_power), axis=1)
    m_power = 10 * np.log10(p + 1.0)

    # save
    if options.loc2d_file is not None:
        outfilename = options.loc2d_file
        np.save(outfilename, pos_list)
        print("[save]", outfilename)

    if options.npy_file is not None:
        outfilename = options.npy_file
        np.save(outfilename, m_power)
        # np.savetxt("music.csv", m_power, delimiter=",")
        print("[save]", outfilename, m_power.shape)

    if options.npy_full_file is not None:
        outfilename = options.npy_full_file
        np.save(outfilename, out_power)
        # np.savetxt("music.csv", m_power, delimiter=",")
        print("[save]", outfilename, power.shape)

    if options.plot_h_file is not None:
        # plot heat map
        ax = sns.heatmap(m_power.transpose(), cbar=False, cmap=cm.Greys)
        sns.plt.axis("off")
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
        outfilename_heat = options.plot_h_file
        sns.plt.savefig(outfilename_heat, bbox_inches="tight", pad_inches=0.0)
        print("[save]", outfilename_heat, m_power.shape)

    if options.plot_hb_file is not None:
        sns.plt.clf()
        sns.heatmap(m_power, cbar=True, cmap=cm.Greys)
        outfilename_heat_bar = options.plot_hb_file
        sns.plt.savefig(outfilename_heat_bar, bbox_inches="tight", pad_inches=0.0)
        sns.plt.clf()
        print("[save]", outfilename_heat_bar, m_power.shape)

if __name__ == "__main__":
    main()
