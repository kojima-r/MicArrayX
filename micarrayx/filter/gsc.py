# -*- coding: utf-8 -*-
from scipy import ceil, complex64, float64, hamming, zeros
from scipy.fftpack import fft
from scipy import ifft
from scipy.io.wavfile import read
import wave
import sys
import numpy as np
import numpy.random as npr
import math

import micarrayx
from hark_tf.read_mat import read_hark_tf
from hark_tf.read_param import read_hark_tf_param
from micarrayx.filter import wiener
from micarrayx.filter.filter_aux import (
    estimate_self_correlation,
    get_beam_vec,
    save_sidelobe,
    get_beam_mat,
)


def get_blocking_mat(n):
    x = np.identity(n)[:-1, :] - np.identity(n)[1:, :]
    return x

def beamforming_ds(tf_config, spec, n_theta=18):
    nch = spec.shape[0]
    nframe = spec.shape[1]
    nfreq_bin = spec.shape[2]
    dtheta=360/n_theta
    theta = [j*dtheta / 180.0 * math.pi for j in range(n_theta)]
    result=[]
    for theta in theta_list:
        index = micarrayx.nearest_direction_index(tf_config, theta)
        a_vec = get_beam_vec(tf_config, index)
        ds_freq = np.zeros((nframe, nfreq_bin), dtype=complex)
        for t in range(nframe):
            for freq_bin in range(nfreq_bin):
                ds_freq[t,freq_bin] = np.dot(a_vec.conj()[freq_bin, :], spec[:, t, freq_bin]) / nch
        result.append(ds_freq)
    return result

def main():
    # argv check
    if len(sys.argv) < 2:
        print(
            "Usage: sim_tf.py <in: tf.zip(HARK2 transfer function file)>",
            file=sys.stderr,
        )
        quit()
    #
    npr.seed(1234)
    tf_filename = sys.argv[1]
    tf_config = read_hark_tf(tf_filename)
    src_theta = 0 / 180.0 * math.pi
    src_index = micarrayx.nearest_direction_index(tf_config, src_theta)
    if not src_index in tf_config["tf"]:
        print(
            "Error: tf index", src_index, "does not exist in TF file", file=sys.stderr
        )
        quit()
    mic_pos = read_hark_tf_param(tf_filename)
    B = get_blocking_mat(8)
    A = get_beam_mat(tf_config, src_index)
    a_vec = get_beam_vec(tf_config, src_index)
    print("# mic positions:", mic_pos)
    ###
    ### apply
    ###
    wav_filename1 = sys.argv[2]
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
    #
    # STFT
    fftLen = 512
    step = 128  # 160
    df = fs1 * 1.0 / fftLen
    # cutoff bin
    min_freq = 0
    max_freq = 10000
    min_freq_bin = int(np.ceil(min_freq / df))
    max_freq_bin = int(np.floor(max_freq / df))
    print("# min freq:", min_freq)
    print("# max freq:", max_freq)
    print("# min fft bin:", min_freq_bin)
    print("# max fft bin:", max_freq_bin)

    #
    win = hamming(fftLen)  # ハミング窓
    spec1 = micarrayx.stft_mch(wav1, win, step)
    # spec1[ch, frame, freq_bin]
    nch = spec1.shape[0]
    nframe = spec1.shape[1]
    nfreq_bin = spec1.shape[2]
    sidelobe_freq_bin = int(np.floor(2000 / df))
    ### DS beamformer & blocked signals
    ds_freq = np.zeros((nframe, nfreq_bin), dtype=complex)
    blocked_freq = np.zeros(
        (spec1.shape[0] - 1, spec1.shape[1], spec1.shape[2]), dtype=complex
    )
    for t in range(spec1.shape[1]):
        for freq_bin in range(spec1.shape[2]):
            blocked_freq[:, t, freq_bin] = B.dot(
                A[freq_bin, :, :].dot(spec1[:, t, freq_bin])
            )
            ds_freq[t, freq_bin] = (
                np.dot(a_vec.conj()[freq_bin, :], spec1[:, t, freq_bin]) / nch
            )
    ds_freq = np.array([ds_freq])
    ### GSC for DS beamformer
    w_a, _, _ = wiener.wiener_filter_freq(blocked_freq, ds_freq)
    y_ds = wiener.apply_filter_freq(blocked_freq, w_a)
    save_sidelobe("sidelobe_ds.png", tf_config, a_vec, sidelobe_freq_bin)
    w_gsc_ds = np.zeros((nfreq_bin, nch), dtype=complex)
    for freq_bin in range(nfreq_bin):
        w_gsc_ds[freq_bin, :] = a_vec[freq_bin, :] - w_a[freq_bin, :].dot(
            B.dot(A[freq_bin, :, :])
        )
    save_sidelobe(
        "sidelobe_gsc_ds.png", tf_config, w_gsc_ds, sidelobe_freq_bin, clear_flag=False
    )
    ### MV beamformer
    # rz=estimate_correlation(spec1,spec1,nframe,1)
    rz = estimate_self_correlation(spec1)
    # rz=np.array([rz])
    w_mv = np.zeros((nfreq_bin, nch), dtype=complex)
    for freq_bin in range(nfreq_bin):
        rz_inv = np.linalg.inv(rz[0, freq_bin, :, :])
        av = a_vec[freq_bin, :].reshape((nch, 1))
        temp = rz_inv.dot(av)
        po = av.T.conj().dot(temp)
        # w[freq_bin,:]=temp.dot(np.linalg.inv(po))
        w_mv[freq_bin, :] = np.squeeze(
            rz_inv.dot(av).dot(np.linalg.inv(av.conj().T.dot(rz_inv).dot(av)))
        )
    mv_freq = wiener.apply_filter_freq(spec1, w_mv)
    # mv_freq=np.array([mv_freq])
    save_sidelobe("sidelobe_mv.png", tf_config, w_mv, sidelobe_freq_bin)
    ### GSC for MV beamformer
    w_a, _, _ = wiener.wiener_filter_freq(blocked_freq, mv_freq)
    y_mv = wiener.apply_filter_freq(blocked_freq, w_a)
    w_gsc_mv = np.zeros((nfreq_bin, nch), dtype=complex)
    for freq_bin in range(nfreq_bin):
        w_gsc_mv[freq_bin, :] = w_mv[freq_bin, :] - w_a[freq_bin, :].dot(
            B.dot(A[freq_bin, :, :])
        )
    save_sidelobe(
        "sidelobe_gsc_mv.png", tf_config, w_gsc_mv, sidelobe_freq_bin, clear_flag=False
    )
    ###
    out_gsc_ds = ds_freq - y_ds
    out_gsc_mv = mv_freq - y_mv
    recons_out_gsc_ds = micarrayx.istft_mch(out_gsc_ds, win, step)
    recons_out_gsc_mv = micarrayx.istft_mch(out_gsc_mv, win, step)
    recons_ds_y = micarrayx.istft_mch(y_ds, win, step)
    recons_mv_y = micarrayx.istft_mch(y_mv, win, step)
    recons_b = micarrayx.istft_mch(blocked_freq, win, step)
    recons_ds = micarrayx.istft_mch(ds_freq, win, step)
    recons_mv = micarrayx.istft_mch(mv_freq, win, step)
    micarrayx.save_mch_wave(recons_mv * 32767.0, "mv.wav")
    micarrayx.save_mch_wave(recons_ds * 32767.0, "ds.wav")
    micarrayx.save_mch_wave(recons_ds_y * 32767.0, "y_ds.wav")
    micarrayx.save_mch_wave(recons_mv_y * 32767.0, "y_mv.wav")
    micarrayx.save_mch_wave(recons_out_gsc_ds * 32767.0, "gsc_ds.wav")
    micarrayx.save_mch_wave(recons_out_gsc_mv * 32767.0, "gsc_mv.wav")
    micarrayx.save_mch_wave(recons_b * 32767.0, "b.wav")

    quit()

if __name__ == "__main__":
    main()
