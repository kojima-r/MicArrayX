# -*- coding: utf-8 -*-

import wave
import array
from matplotlib import pylab as pl
import sys
import numpy as np
import numpy.random as npr
import math

from hark_tf.read_mat import read_hark_tf
from hark_tf.read_param import read_hark_tf_param
import simmch
from sim_tf import apply_tf

from optparse import OptionParser

def main():
    usage = "usage: %s tf [options] <in: src.wav> <out: dest.wav>" % sys.argv[0]
    parser = OptionParser()
    parser.add_option(
        "-t",
        "--tf",
        dest="tf",
        help="tf.zip(HARK2 transfer function file>",
        default=None,
        type=str,
        metavar="TF",
    )

    parser.add_option(
        "-d",
        "--direction",
        dest="direction",
        help="arrival direction of sound (degree)",
        default=0,
        type=float,
        metavar="DIRECTION",
    )

    parser.add_option(
        "-c",
        "--channel",
        dest="channel",
        help="target channel of input sound (>=0)",
        default=0,
        type=int,
        metavar="CH",
    )

    parser.add_option(
        "-V",
        "--volume",
        dest="volume",
        help="volume of input sound (0<=v<=1)",
        default=1,
        type=float,
        metavar="VOL",
    )

    parser.add_option(
        "-N",
        "--noise",
        dest="noise",
        help="noise amplitude",
        default=0,
        type=float,
        metavar="N",
    )

    (options, args) = parser.parse_args()

    # argv check
    if len(args) < 1:
        quit()
    #
    npr.seed(1234)
    tf_filename = options.tf
    tf_config = read_hark_tf(tf_filename)
    target_ch = options.channel
    src_theta = options.direction / 180.0 * math.pi
    src_index = simmch.nearest_direction_index(tf_config, src_theta)
    src_volume = options.volume
    # output_filename=args[1]
    if not src_index in tf_config["tf"]:
        print(
            "Error: tf index", src_index, "does not exist in TF file", file=sys.stderr
        )
        quit()
    mic_pos = read_hark_tf_param(tf_filename)
    print("# mic positions:", mic_pos)
    wav_filename = args[0]
    wav_data = simmch.read_mch_wave(wav_filename)
    wav = wav_data["wav"] / 32767.0
    fs = wav_data["framerate"]
    nch = wav_data["nchannels"]
    # print info
    print("# channel num : ", nch)
    print("# sample size : ", wav.shape)
    print("# sampling rate : ", fs)
    print("# sec : ", wav_data["duration"])

    # reading data
    # ch, frame, bin
    mono_wavdata = wav[target_ch]
    data = mono_wavdata

    fftLen = 512
    step = fftLen / 4
    # apply transfer function
    mch_wavdata = apply_tf(data, fftLen, step, tf_config, src_index, noise_amp=1)
    mch_wavdata = mch_wavdata * src_volume
    print(mch_wavdata.shape)

if __name__ == "__main__":
    main()
