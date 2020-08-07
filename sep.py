# -*- coding: utf-8 -*-
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import wave
import array
import sys
import numpy as np
import numpy.random as npr
import math
from optparse import OptionParser
import simmch


def zcc(x):
    counter = 0
    for i in range(x.shape[0] - 1):
        if x[i] >= 0 and x[i + 1] < 0:
            counter += 1
    return counter


if __name__ == "__main__":
    usage = "usage: %s [options] <in: src.wav>" % sys.argv[0]
    parser = OptionParser()
    parser.add_option(
        "-o",
        "--output",
        dest="output_file",
        help="output file",
        default=None,
        type=str,
        metavar="FILE",
    )

    parser.add_option(
        "-A",
        "--amp",
        dest="thresh_amp",
        help="threshold of amplitude",
        default=None,
        type=float,
        metavar="AMP",
    )

    parser.add_option(
        "-P",
        "--pow",
        dest="thresh_power",
        help="threshold of power",
        default=None,
        type=float,
        metavar="POWER",
    )

    parser.add_option(
        "-Z",
        "--zerocross",
        dest="thresh_zcc",
        help="threshold of zero cross count",
        default=None,
        type=float,
        metavar="COUNT",
    )

    parser.add_option(
        "-S",
        "--segment_size",
        dest="seg_size",
        help="threshold of minimum segment size (second)",
        default=None,
        type=float,
        metavar="DURATION",
    )

    parser.add_option(
        "-c",
        "--ch",
        dest="target_channel",
        help="target channel of input wav",
        default=0,
        type=int,
        metavar="CH",
    )

    parser.add_option(
        "-p",
        "--plot",
        dest="plot_output",
        help="output filename for plotting data",
        default=None,
        type=str,
        metavar="FILE",
    )

    (options, args) = parser.parse_args()

    # argv check
    if len(args) < 1:
        quit()
    #
    thresh_amp = options.thresh_amp
    thresh_pow = options.thresh_power
    thresh_zcc = options.thresh_zcc
    thresh_seg_size = options.seg_size
    output_filename = options.output_file
    data = []
    print("... reading .wav files")
    wav_filename = args[0]
    wav_data = simmch.read_mch_wave(wav_filename)
    wav = wav_data["wav"]
    fs = wav_data["framerate"]
    nch = wav_data["nchannels"]
    data = (wav, fs, nch, wav_filename)
    print("... checking segments")
    length = 512
    step = length / 4
    win = [1.0] * length
    ch = options.target_channel
    if ch >= nch:
        print("[ERROR] target channel (%d) does not exist" % ch, file=sys.stderr)
        quit()
    x = simmch.apply_window(wav[ch], win, step)
    nframe = x.shape[0]
    x_power = [np.mean(x[m, :] ** 2) for m in range(nframe)]
    x_amp = [np.max(abs(x[m, :])) for m in range(nframe)]
    x_zcc = [zcc(x[m, :]) for m in range(nframe)]
    start_frame = 0
    segment_enabled_flag = False
    frame_segs = []
    for m in range(nframe):
        if segment_enabled_flag:
            if thresh_pow != None and x_power[m] < thresh_pow:
                segment_enabled_flag = False
            if thresh_amp != None and x_amp[m] < thresh_amp:
                segment_enabled_flag = False
            if thresh_zcc != None and x_zcc[m] < thresh_zcc:
                segment_enabled_flag = False
            if not segment_enabled_flag:
                frame_segs.append((start_fram, m))
        else:
            if thresh_pow != None and x_power[m] >= thresh_pow:
                segment_enabled_flag = True
            if thresh_amp != None and x_amp[m] >= thresh_amp:
                segment_enabled_flag = True
            if thresh_zcc != None and x_zcc[m] >= thresh_zcc:
                segment_enabled_flag = True
            if segment_enabled_flag:
                start_fram = m
    print("... detected segments by frame-based thresholds")
    sample_segs = [(seg[0] * step, seg[1] * step + length) for seg in frame_segs]
    print(sample_segs)
    print("... detected segments")
    filtered_segs = [
        seg for seg in sample_segs if seg[1] - seg[0] > thresh_seg_size * fs
    ]
    print(filtered_segs)
    for seg_id, s in enumerate(filtered_segs):
        w = wav[:, s[0] : s[1]]
        # save data
        if options.output_file != None:
            o = options.output_file % seg_id
            print("[save]", o)
            simmch.save_mch_wave(w, o)
    #
    plt.subplot(4, 1, 1)
    disable_xy_label = False
    if disable_xy_label:
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
    plt.xlabel("time [second]")
    plt.ylabel("wav")
    plt.plot(wav[ch])
    #
    plt.subplot(4, 1, 2)
    disable_xy_label = False
    if disable_xy_label:
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
    plt.xlabel("time [second]")
    plt.ylabel("amp")
    plt.plot(x_amp)

    #
    plt.subplot(4, 1, 3)
    disable_xy_label = False
    if disable_xy_label:
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
    plt.xlabel("time [second]")
    plt.ylabel("power")
    plt.plot(x_power)
    #
    plt.subplot(4, 1, 4)
    disable_xy_label = False
    if disable_xy_label:
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
    plt.xlabel("time [second]")
    plt.ylabel("ZCC")
    plt.plot(x_zcc)

    try:
        plt.savefig(options.plot_output, dpi=72)
    except:
        print("[WARN] too short (plot fail)")
