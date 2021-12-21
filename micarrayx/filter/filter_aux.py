import sys
import numpy as np
import numpy.random as npr
from scipy import hamming, interpolate
import scipy
import math

# import matplotlib
# matplotlib.use('tkagg')
from matplotlib import pylab as plt

def estimate_correlation_r(spec1, spec2, win_size, step):
    # ch,frame,spec -> frame,spec,ch
    n_ch1 = spec1.shape[0]
    n_ch2 = spec2.shape[0]
    n_frame = spec1.shape[1]
    n_bin = spec1.shape[2]
    corr = np.zeros((n_frame, n_bin, n_ch1, n_ch2), dtype=complex)

    # out_corr: frame,spec,ch1,ch2
    for i in range(n_ch1):
        for j in range(n_ch2):
            corr[:, :, i, j] = (
                spec1[i]
                * spec2[j].conj()
                / (np.absolute(spec1[i]) * np.absolute(spec2[j]))
            )
    now_frame = 0
    out = []
    while now_frame + win_size <= n_frame:
        o = np.mean(corr[now_frame : now_frame + win_size], axis=0)
        np.max(o)
        out.append(o)
        now_frame += step

    return np.array(out)


def estimate_correlation(spec1, spec2, win_size, step):
    # ch,frame,spec
    n_ch1 = spec1.shape[0]
    n_ch2 = spec2.shape[0]
    n_frame = spec1.shape[1]
    n_bin = spec1.shape[2]
    corr = np.zeros((n_frame, n_bin, n_ch1, n_ch2), dtype=complex)

    # out_corr: frame,spec,ch1,ch2
    now_frame = 0
    out = []
    while now_frame + win_size <= n_frame:
        corr = np.zeros((n_bin, n_ch1, n_ch2), dtype=complex)
        s1 = np.transpose(spec1[:, now_frame : now_frame + win_size, :], (1, 0, 2))
        s2 = np.transpose(spec2[:, now_frame : now_frame + win_size, :], (1, 0, 2))
        es1 = np.mean(s1, axis=0)
        es2 = np.mean(s2, axis=0)
        # t,ch,spec
        es1 = np.tile(es1, (win_size, 1, 1))
        es2 = np.tile(es2, (win_size, 1, 1))
        ss1 = s1 - es1
        ss2 = s2 - es2
        for i in range(n_ch1):
            for j in range(n_ch2):
                cij = np.sum(ss1[:, i, :] * ss2[:, j, :].conj(), axis=0) / (
                    win_size - 1
                )
                cii = np.sum(ss1[:, i, :] * ss1[:, i, :].conj(), axis=0) / (
                    win_size - 1
                )
                cjj = np.sum(ss2[:, j, :] * ss2[:, j, :].conj(), axis=0) / (
                    win_size - 1
                )
                # print cij[64]/np.sqrt(cii[64]*cjj[64])
                corr[:, i, j] = cij / np.sqrt(cii * cjj)
        out.append(corr)
        # repmat(es2,(1,win_size,1))
        # o=np.mean(corr[now_frame:now_frame+win_size],axis=0)
        now_frame += step

    return np.array(out)


def estimate_self_correlation(spec1):
    # ch,frame,spec -> frame,spec,ch
    n_ch1 = spec1.shape[0]
    n_frame = spec1.shape[1]
    n_bin = spec1.shape[2]
    corr = np.zeros((1, n_bin, n_ch1, n_ch1), dtype=complex)

    # out_corr: frame,spec,ch1,ch2
    for fbin in range(n_bin):
        vec_x = [spec1[i, :, fbin] for i in range(n_ch1)]
        corr[0, fbin, :, :] = np.corrcoef(vec_x)
    return corr


def get_beam_vec(tf_config, src_index):
    tf = tf_config["tf"][src_index]
    # print "# position:",tf["position"]
    mat = []
    for mic_index in range(tf["mat"].shape[0]):
        tf_mono = tf["mat"][mic_index]
        # print "# tf spectrogram:",tf_mono.shape
        mat.append(tf_mono)
    m = np.array(mat).T
    # mm=m/np.tile(np.sum(m.conj()*m,axis=1),(tf["mat"].shape[0],1)).T
    return m


def get_all_sidelobe(tf_config, w):
    sidelobe = {}
    for src_index in range(len(tf_config["tf"])):
        tf = tf_config["tf"][src_index]
        pos = tf["position"]
        th = math.atan2(pos[1], pos[0])  # -pi ~ pi
        a_vec = get_beam_vec(tf_config, src_index)
        g = np.sum(a_vec.conj() * w, axis=1)
        sidelobe[th] = g
    return sidelobe


def get_sidelobe(tf_config, w, freq_bin):
    sidelobes = get_all_sidelobe(tf_config, w)
    s = {}
    for th, g in list(sidelobes.items()):
        s[th] = g[freq_bin]
    return s


def get_beam_mat(tf_config, src_index):
    m = get_beam_vec(tf_config, src_index)
    x = np.zeros((m.shape[0], m.shape[1], m.shape[1]), dtype=complex)
    for fs in range(m.shape[0]):
        x[fs, :, :] = np.diag(m[fs])
    return x


def save_sidelobe(filename, tf_config, w, sf_bin, clear_flag=True):
    sidelobe = get_sidelobe(tf_config, w, sf_bin)
    thetas = []
    gains = []
    for theta, gain in list(sidelobe.items()):
        thetas.append(theta)
        gains.append(20 * np.log10(np.absolute(gain)))
    ## save
    if clear_flag:
        plt.clf()
    plt.plot(thetas, gains, "o")
    plt.savefig(filename)
