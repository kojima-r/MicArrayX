# -*- coding: utf-8 -*-
""" command: micarrayx-make-noise
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
from micarrayx import nearest_direction_index

from optparse import OptionParser


def rand_noise(x):
    rad = npr.rand() * 2 * math.pi
    return math.cos(rad) + 1j * math.sin(rad)


def make_white_noise_freq(nch, length, fftLen, step):
    """白色ノイズの作成（周波数領域）

    関数についての説明文
    Args:
        nch (int):    チャンネル数
        length (int): STFTパラメータ：サンプル数
        fftLen (int): STFTパラメータ：FFTの長さ
        step (int):   STFTパラメータ：

    Returns:
        ndarray: ホワイトノイズの結果
    
    Examples:
        >>> nsamples=16000
        >>> fftLen=512
        >>> step=256
        >>> length = ((nsamples - (fftLen - step)) - 1) / step + 1
        >>> wav_freq=make_white_noise_freq(nch=1, length=length, fftLen=fftLen, step=step)
        >>> wav_freq.shape

    """
    # stft length <-> samples
    src_volume = 1
    data = np.zeros((nch, int(length), fftLen // 2 + 1), dtype=complex64)
    v_make_noise = np.vectorize(rand_noise)
    data = v_make_noise(data)

    # win = hamming(fftLen) # ハミング窓
    win = np.array([1.0] * fftLen)
    out_data = []
    for mic_index in range(data.shape[0]):
        spec = data[mic_index]
        full_spec = micarrayx.make_full_spectrogram(spec)
        # s_sum=np.mean(np.abs(full_spec)**2,axis=1)
        # print "[CHECK] power(spec/frame):",np.mean(s_sum)
        out_data.append(full_spec)
    # concat waves
    mch_data = np.array(out_data)
    return mch_data


def make_white_noise(nch, length, fftLen, step):
    """白色ノイズの作成

    関数についての説明文
    Args:
        nch (int):    チャンネル数
        length (int): STFTパラメータ：サンプル数
        fftLen (int): STFTパラメータ：FFTの長さ
        step (int):   STFTパラメータ：

    Returns:
        ndarray: ホワイトノイズの結果
    
    Examples:
        >>> nsamples=16000
        >>> fftLen=512
        >>> step=256
        >>> length = ((nsamples - (fftLen - step)) - 1) / step + 1
        >>> wav=make_white_noise(nch=1, length=length, fftLen=fftLen, step=step)
        >>> wav.shape

    """
    # stft length <-> samples
    src_volume = 1
    data = make_white_noise_freq(nch, length, fftLen, step)

    # win = hamming(fftLen) # ハミング窓
    win = np.array([1.0] * fftLen)
    out_wavdata = []
    for mic_index in range(data.shape[0]):
        spec = data[mic_index]
        ### iSTFT
        resyn_data = micarrayx.istft(spec, win, step)
        # x=micarrayx.apply_window(resyn_data, win, step)
        # w_sum=np.sum(x**2,axis=1)
        # print "[CHECK] power(x/frame):",np.mean(w_sum)
        out_wavdata.append(resyn_data)
    # concat waves
    mch_wavdata = np.vstack(out_wavdata)
    amp = np.max(np.abs(mch_wavdata))
    return mch_wavdata / amp


def main():
    usage = "usage: %s [options] <out: dest.wav>" % sys.argv[0]
    parser = OptionParser(usage)
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
        "--duration",
        dest="duration",
        help="duration of sound (sec)",
        default=None,
        type=float,
        metavar="D",
    )

    parser.add_option(
        "-s",
        "--samples",
        dest="samples",
        help="duration of sound (samples)",
        default=None,
        type=int,
        metavar="D",
    )

    parser.add_option(
        "-f",
        "--frames",
        dest="frames",
        help="duration of sound (frames)",
        default=None,
        type=int,
        metavar="D",
    )

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
        default=1.0,
        type=float,
        metavar="AMP",
    )

    parser.add_option(
        "-T",
        "--template",
        dest="template",
        help="template wav file",
        default=None,
        type=str,
        metavar="FILE",
    )

    (options, args) = parser.parse_args()

    # argv check
    if len(args) < 1:
        parser.print_help()
        quit()
    #
    npr.seed(1234)
    output_filename = args[0]
    # save data
    nch = options.channel
    fftLen = 512
    step = fftLen / 4
    nsamples = None
    if options.samples != None:
        nsamples = options.samples
        length = ((nsamples - (fftLen - step)) - 1) / step + 1
    elif options.frames != None:
        length = options.frames
        nsamples = length * step + (fftLen - step)
    elif options.duration != None:
        nsamples = options.samplingrate * options.duration
        length = ((nsamples - (fftLen - step)) - 1) / step + 1
    elif options.template != None:
        wav_filename = options.template
        print("... reading", wav_filename)
        wav_data = micarrayx.read_mch_wave(wav_filename)
        nsamples = wav_data["nframes"]
        nch = wav_data["nchannels"]
        length = ((nsamples - (fftLen - step)) - 1) / step + 1
    else:
        print(
            "[ERROR] unknown duration (please indicate --duration, --samples, or --frames)",
            file=sys.stderr,
        )
        quit()

    print("#channels:", nch)
    print("#frames:", length)
    print("#samples:", nsamples)
    print("window size:", fftLen)
    mch_wavdata = make_white_noise(nch, length, fftLen, step)

    g = 32767.0 * options.amp
    print("[INFO] gain:", g)
    mch_wavdata = mch_wavdata * g
    micarrayx.save_mch_wave(
        mch_wavdata, output_filename, sample_width=2, framerate=options.samplingrate
    )

if __name__ == "__main__":
    main()
