# -*- coding: utf-8 -*-

import wave
import array
import sys
import numpy as np
import numpy.random as npr
import math
from optparse import OptionParser
import micarrayx

def main():
    usage = "usage: %s tf [options] <in: src.wav>" % sys.argv[0]
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
        "-V",
        "--volume",
        dest="volume",
        help="volumes of input sound (0<=v<=1)",
        default=1.0,
        type=str,
        metavar="VOL",
    )

    (options, args) = parser.parse_args()

    # argv check
    if len(args) < 1:
        quit()
    #
    npr.seed(1234)
    src_volume = options.volume
    output_filename = options.output_file
    data = []
    #
    wav_filename = args[0]
    print("... reading", wav_filename)
    wav_data = micarrayx.read_mch_wave(wav_filename)
    wav = wav_data["wav"]
    fs = wav_data["framerate"]
    nch = wav_data["nchannels"]
    amp = np.max(np.abs(wav))
    print("[INFO] max amplitude:", amp)
    g = 32767.0 / amp * src_volume
    print("[INFO] gain:", g)
    wav = wav * g
    # save data
    if output_filename != None:
        micarrayx.save_mch_wave(wav, output_filename)

if __name__ == "__main__":
    main()
