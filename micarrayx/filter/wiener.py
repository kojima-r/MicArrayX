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

from optparse import OptionParser
import micarrayx
from hark_tf.read_mat import read_hark_tf
from hark_tf.read_param import read_hark_tf_param

from micarrayx.filter.filter_aux import estimate_correlation, estimate_self_correlation, save_sidelobe


def apply_filter_freq(spec1, w):
    wh = w.conj().T

    axis_ch = 0
    if len(wh.shape) == 3:
        # wh: nch(output),nch, freq_bin
        axis_ch = 1
        out_spec = np.zeros(
            (spec1.shape[1], wh.shape[0], spec1.shape[2]), dtype=complex
        )
    else:
        # wh: nch, freq_bin
        out_spec = np.zeros((spec1.shape[1], spec1.shape[2]), dtype=complex)
    for t in range(spec1.shape[1]):
        oo = wh * spec1[:, t, :]
        out_spec[t, :] = oo.sum(axis=axis_ch)
    if len(wh.shape) == 3:
        return out_spec.transpose(1, 0, 2)
    else:
        return np.array([out_spec])


def wiener_filter_freq(spec1, spec2, win_size=None, r_step=1):
    spec1_temp = spec1
    spec2_temp = spec2
    nframe = spec1.shape[1]
    if spec1.shape[1] != spec2.shape[1]:
        nframe1 = spec1.shape[1]
        nframe2 = spec2.shape[1]
        nframe = min(nframe1, nframe2)
        spec1_temp = spec1[:, 0:nframe, :]
        spec2_temp = spec2[:, 0:nframe, :]
    if win_size == None:
        win_size = nframe
    rz = estimate_correlation(spec1_temp, spec1_temp, win_size, r_step)
    rzd = estimate_correlation(spec1_temp, spec2_temp, win_size, r_step)
    rz = np.squeeze(rz)
    rzd = np.squeeze(rzd)
    w = np.zeros(rzd.shape, dtype=complex)
    for i in range(rzd.shape[0]):
        # print np.linalg.inv(rz)#+np.identity(rz.shape[1]))
        # Ax=b
        w[i, :] = np.linalg.solve(rz[i, :, :], rzd[i, :])
    return w, rz, rzd


def wiener_filter_eigen(spec1, spec2, win_size=None, r_step=1):
    spec1_temp = spec1
    spec2_temp = spec2
    nframe = spec1.shape[1]
    if spec1.shape[1] != spec2.shape[1]:
        nframe1 = spec1.shape[1]
        nframe2 = spec2.shape[1]
        nframe = min(nframe1, nframe2)
        spec1_temp = spec1[:, 0:nframe, :]
        spec2_temp = spec2[:, 0:nframe, :]
    if win_size == None:
        win_size = nframe

    r1 = estimate_correlation(spec1_temp, spec1_temp, win_size, r_step)
    r2 = estimate_correlation(spec2_temp, spec2_temp, win_size, r_step)
    # r1=estimate_self_correlation(spec1_temp)
    # r2=estimate_self_correlation(spec2_temp)
    out_w = np.zeros(r1.shape, dtype=complex)
    for frame in range(r1.shape[0]):
        for freq_bin in range(r1.shape[1]):
            # a   vr[:,i] = w[i]        b   vr[:,i]
            rz = r1[frame, freq_bin, :, :]
            k = r2[frame, freq_bin, :, :]
            w, vr = scipy.linalg.eig(a=rz, b=k)
            eigen_id = np.argsort(w)[::-1]
            eigen_values = w[eigen_id]
            eigen_vecs = vr[:, eigen_id]
            v1_inv = np.linalg.inv(eigen_vecs.conj().T)
            v1 = eigen_vecs.conj().T
            v2 = eigen_vecs
            # i=0
            # print k
            # print "====="
            # print rz.dot(eigen_vecs[:,i])
            # print w[i]*k.dot(eigen_vecs[:,i])
            # print "====="
            # print eigen_values
            # print (v1.dot(rz).dot(v2))
            # print (v1.dot(k).dot(v2))
            l = np.diagonal(v1.dot(rz).dot(v2))
            s = np.diagonal(v1.dot(k).dot(v2))
            one = np.ones_like(l)
            g = one - s / l
            G = np.diag(g)
            out_w[frame, freq_bin, :, :] = v1_inv.dot(G.dot(v1))
    return out_w


def apply_filter_eigen(spec1, w):
    out_spec = np.zeros((spec1.shape[0], spec1.shape[1], spec1.shape[2]), dtype=complex)
    for frame in range(spec1.shape[1]):
        for freq_bin in range(spec1.shape[2]):
            oo = w[0, freq_bin, :].dot(spec1[:, frame, freq_bin])
            out_spec[:, frame, freq_bin] = oo
    return out_spec


def main():
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
        "-e", "--noise", dest="noise", help="", default=None, type=str, metavar="FILE"
    )
    (options, args) = parser.parse_args()

    # argv check
    if len(args) < 2:
        print("Usage: music.py <in: src.wav> <in: desired.wav>", file=sys.stderr)
        quit()
    # read tf
    npr.seed(1234)

    # read wav (src)
    wav_filename1 = args[0]
    print("... reading", wav_filename1)
    wav_data1 = micarrayx.read_mch_wave(wav_filename1)
    wav1 = wav_data1["wav"] / 32767.0
    fs1 = wav_data1["framerate"]
    nch1 = wav_data1["nchannels"]
    # print info
    print("# channel num : ", nch1)
    print("# sample size : ", wav1.shape)
    print("# sampling rate : ", fs1)
    print("# sec : ", wav_data1["duration"])

    # read wav (desired)
    wav_data_list = []
    for wav_filename2 in args[1:]:
        print("... reading", wav_filename2)
        wav_data2 = micarrayx.read_mch_wave(wav_filename2)
        wav2 = wav_data2["wav"] / 32767.0
        fs2 = wav_data2["framerate"]
        nch2 = wav_data2["nchannels"]
        # print info
        print("# channel num : ", nch2)
        print("# sample size : ", wav2.shape)
        print("# sampling rate : ", fs2)
        print("# sec : ", wav_data2["duration"])
        wav_data_list.append(wav2)
    wav2 = np.vstack(wav_data_list)
    print(wav2.shape)
    # reading data
    fftLen = 512
    step = 160  # fftLen / 4
    df = fs1 * 1.0 / fftLen
    # cutoff bin
    min_freq = 0
    max_freq = 10000
    min_freq_bin = int(np.ceil(min_freq / df))
    max_freq_bin = int(np.floor(max_freq / df))
    sidelobe_freq_bin = int(np.floor(2000 / df))
    print("# min freq:", min_freq)
    print("# max freq:", max_freq)
    print("# min fft bin:", min_freq_bin)
    print("# max fft bin:", max_freq_bin)

    # STFT
    win = hamming(fftLen)  # ハミング窓
    spec1 = micarrayx.stft_mch(wav1, win, step)
    spec2 = micarrayx.stft_mch(wav2, win, step)
    ##
    ##
    nframe1 = spec1.shape[1]
    nframe2 = spec2.shape[1]
    nframe = min(nframe1, nframe2)
    spec1_temp = spec1[:, 0:nframe, min_freq_bin:max_freq_bin]
    spec2_temp = spec2[:, 0:nframe, min_freq_bin:max_freq_bin]

    if options.noise is not None:
        w = wiener_filter_eigen(spec1_temp, spec2_temp, win_size=nframe, r_step=1)
        print("# filter:", w.shape)
        out_spec = apply_filter_eigen(spec1_temp, w)
        # ISTFT
        recons = micarrayx.istft_mch(out_spec, win, step)
        micarrayx.save_mch_wave(recons * 32767.0, "recons_eigen.wav")
        quit()
    # print spec1_temp.shape
    # print spec2_temp.shape
    # win_size=50
    # spec[ch, frame, freq_bin]
    # w[ch2,ch1]
    w, _, _ = wiener_filter_freq(spec1_temp, spec2_temp, win_size=nframe, r_step=1)
    print("# filter:", w.shape)
    if options.tf is not None:
        tf_config = read_hark_tf(options.tf)
        if len(w.shape) == 3:
            for i in range(len(w.shape)):
                save_sidelobe(
                    "sidelobe_wiener%i.png" % (i + 1),
                    tf_config,
                    w[:, :, i],
                    sidelobe_freq_bin,
                )
        else:
            save_sidelobe("sidelobe_wiener.png", tf_config, w, sidelobe_freq_bin)
    # filter
    out_spec = apply_filter_freq(spec1_temp, w)
    # ISTFT
    recons = micarrayx.istft_mch(out_spec, win, step)

    # recons.reshape((recons.shape[0],1))
    micarrayx.save_mch_wave(recons * 32767.0, "recons_wiener.wav")

if __name__ == "__main__":
    main()
