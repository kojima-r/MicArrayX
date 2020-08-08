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
    usage = "usage: %s tf [options] <in: src.wav ...>" % sys.argv[0]
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
    parser.add_option(
        "-N",
        "--normalize",
        dest="normalization",
        help="volumes of input sound (0<=v<=1), comma separated",
        default=None,
        type=str,
        metavar="G",
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
    print("... reading .wav files")
    for wav_filename in args:
        print(wav_filename)
        wav_data = micarrayx.read_mch_wave(wav_filename)
        wav = wav_data["wav"]
        fs = wav_data["framerate"]
        nch = wav_data["nchannels"]
        data.append((wav, fs, nch, wav_filename))
    print("... checking")
    wavdata = data[0][0]
    fs = data[0][1]
    nch = data[0][2]
    wav_length = 0
    for wav_info in data:
        if nch != wav_info[2]:
            print(
                "[ERROR] #channel error:",
                nch,
                "!=",
                wav_info[2],
                "(",
                wav_info[3],
                ")",
                file=sys.stderr,
            )
        if fs != wav_info[1]:
            print(
                "[ERROR] sampling rate error",
                fs,
                "!=",
                wav_info[1],
                "(",
                wav_info[3],
                ")",
                file=sys.stderr,
            )
        if wav_length < wav_info[0].shape[1]:
            wav_length = wav_info[0].shape[1]
    print("... mixing")
    mix_wavdata = np.zeros((nch, wav_length), dtype="float")
    normalization = None
    if options.normalization is not None:
        arr = options.normalization.split(",")
        normalization = list(map(float, arr))

    for i, wav_info in enumerate(data):
        l = wav_info[0].shape[1]
        wav = wav_info[0].astype("float")
        if normalization is not None:
            v = normalization[i]
            amp = np.max(np.abs(wav))
            print("[INFO] max amplitude:", amp, "->", v)
            wav = wav * v / amp
        mix_wavdata[:, :l] += wav
    amp = np.max(np.abs(mix_wavdata))
    print("[INFO] max amplitude:", amp)
    g = 32767.0 / amp * src_volume
    print("[INFO] scaling gain:", g)
    mix_wavdata *= g
    # save data
    if output_filename != None:
        micarrayx.save_mch_wave(mix_wavdata, output_filename)

if __name__ == "__main__":
    main()
