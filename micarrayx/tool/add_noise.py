# -*- coding: utf-8 -*-
""" command: micarrayx-add-noise
"""
from scipy import ceil, complex64, float64, hamming, zeros
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
from micarrayx.simulator.sim_tf import apply_tf
from micarrayx.tool.make_noise import make_white_noise_freq, make_white_noise
from micarrayx import nearest_direction_index

from optparse import OptionParser



def main():
    usage = "usage: %s [options] <in: src.wav> <out: dest.wav>" % sys.argv[0]
    parser = OptionParser(usage)
    
    parser.add_option(
        "-r",
        "--samplingrate",
        dest="samplingrate",
        help="sampling rate",
        default=16000,
        type=int,
        metavar="R",
    )

    parser.add_option(
        "-c",
        "--channel",
        dest="channel",
        help="target channel of input sound (>=0)",
        default=1,
        type=int,
        metavar="CH",
    )

    parser.add_option(
        "-A",
        "--amplitude",
        dest="amp",
        help="amplitude of output sound (0<=v<=1)",
        default=None,
        type=float,
        metavar="AMP",
    )

    parser.add_option(
        "-N",
        "--noise_ratio",
        dest="noise_ratio",
        help="noise ratio (0<=v<=1)",
        default=1.0,
        type=float,
        metavar="AMP",
    )



    (options, args) = parser.parse_args()

    # argv check
    if len(args) < 1:
        parser.print_help()
        quit()
    #
    npr.seed(1234)
    input_filename = args[0]
    output_filename = args[1]
    # save data
    fftLen = 512
    step = fftLen / 4
    
    print("... reading", input_filename)
    wav_data = micarrayx.read_mch_wave(input_filename)
    nsamples = wav_data["nframes"]
    nch = wav_data["nchannels"]
    wav = wav_data["wav"]
    if options.amp:
        amp = np.max(np.abs(wav))
        print("[INFO] max amplitude:", amp)
        g = 32767.0 / amp * options.amp
        print("[INFO] gain:", g)
        src_wav = wav * g
    else:
        src_wav = wav
    #
    length = ((nsamples - (fftLen - step)) - 1) / step + 1
    
    print("#channels:", nch)
    print("#frames:", length)
    print("#samples:", nsamples)
    print("window size:", fftLen)
    noise_wav = make_white_noise(nch, length, fftLen, step)
    amp = np.max(np.abs(noise_wav))
    print("[INFO] max noise amplitude:", amp)
    g = 32767.0 / amp
    print("[INFO] gain:", g)
    noise_wav = noise_wav * g

    print("... mixing")
    l=min([noise_wav.shape[1],src_wav.shape[1]])
    # noise ratio = A_n / A_s
    # SN = -20 log noise_ratio
    out_wav=(noise_wav[:,:l]*options.noise_ratio+src_wav[:,:l])/(1.0+options.noise_ratio)
    
    # save data
    if output_filename != None:
        micarrayx.save_mch_wave(out_wav, output_filename, framerate=wav_data["framerate"],sample_width=wav_data["sample_width"])

if __name__ == "__main__":
    main()
