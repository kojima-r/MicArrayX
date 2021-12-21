# -*- coding: utf-8 -*-
""" command: micarrayx-localize
"""
import sys
import numpy as np
from scipy import hamming, interpolate
import json
import argparse

import matplotlib

matplotlib.use("Agg")
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from hark_tf.read_mat import read_hark_tf
from hark_tf.read_param import read_hark_tf_param
import micarrayx
from micarrayx.localization import music


def detect_island(vec):
    """ 入力ベクトルの0以上の部分でまとまり（island）を検出する

    
    >>> e.g.: [  1,  2,  3,  2,  -1,  -3, -1,  2,  3]
    >>> =>  [[(0,1),(1,2),(2,3),(3,2)], [(7,2),(8,3)]]

    Args:
        vec (ndarray): 入力信号(一次元信号)

    Returns:
        List: island のリスト

    """

    o = []
    out = []
    for i, el in enumerate(vec):
        if el > 0:
            o.append((i, el))
        else:
            if len(o) > 0:
                out.append(o)
                o = []
    if len(o) > 0:
        if vec[0] > 0:
            if len(out) > 0:
                out[0] = o + out[0]
            else:
                out.append(o)
        else:
            out.append(o)
    return out


def detect_all_peaks(vec):
    """ 入力ベクトルのピークを検出する

    
    >>> e.g.: [  1,  2,  3,  2,  -1,  -3, -1,  2,  3]
    >>> =>  [[([2],3),([8],3)]]

    Args:
        vec (ndarray): 入力信号(一次元信号)

    Returns:
        List: ピーク点 のリスト

    """
    islands = detect_island(vec)
    peaks = []
    for el in islands:
        max_i = None
        max_v = None
        for e in el:
            if e[1] == max_v:
                max_i.append(e[0])
            elif max_v is None or e[1] > max_v:
                max_v = e[1]
                max_i = [e[0]]
        peaks.append((max_i, max_v))
    return peaks


def detect_peak(vec):
    """ 入力ベクトルのピークを検出する(同じ点が複数ある場合に中央をとる)

    
    >>> e.g.: [  1,  2,  3,  2,  -1,  -3, -1,  2,  3]
    >>> =>  [[(2,3),(8,3)]]

    Args:
        vec (ndarray): 入力信号(一次元信号)

    Returns:
        List: ピーク点 のリスト

    """
    peaks = []
    all_peaks = detect_all_peaks(vec)
    for el in all_peaks: #同じ点が複数場合
        c = len(el[0]) // 2
        peaks.append(el[0][c])
    return peaks


def main():
    # argv check
    parser = argparse.ArgumentParser(
        description="applying the MUSIC method to am-ch wave file"
    )
    #### option for the MUSIC method
    parser.add_argument(
        "tf_filename",
        metavar="TF_FILE",
        type=str,
        help="HARK2.0 transfer function file (.zip)",
    )
    parser.add_argument(
        "wav_filename", metavar="WAV_FILE", type=str, help="target wav file"
    )
    parser.add_argument(
        "--normalize_factor",
        metavar="V",
        type=int,
        default=32768.0,
        help="normalize factor for the given wave data(default=sugned 16bit)",
    )
    parser.add_argument(
        "--stft_win_size",
        metavar="S",
        type=int,
        default=512,
        help="window sise for STFT",
    )
    parser.add_argument(
        "--stft_step",
        metavar="S",
        type=int,
        default=128,
        help="advance step size for STFT (c.f. overlap=fftLen-step)",
    )
    parser.add_argument(
        "--min_freq",
        metavar="F",
        type=float,
        default=300,
        help="minimum frequency of MUSIC spectrogram (Hz)",
    )
    parser.add_argument(
        "--max_freq",
        metavar="F",
        type=float,
        default=8000,
        help="maximum frequency of MUSIC spectrogram (Hz)",
    )
    parser.add_argument(
        "--music_win_size",
        metavar="S",
        type=int,
        default=50,
        help="block size to compute a correlation matrix for the MUSIC method (frame)",
    )
    parser.add_argument(
        "--music_step",
        metavar="S",
        type=int,
        default=50,
        help="advanced step block size (i.e. frequency of computing MUSIC spectrum) (frame)",
    )
    parser.add_argument(
        "--music_src_num",
        metavar="N",
        type=int,
        default=3,
        help="the number of sound source candidates  (i.e. # of dimensions of the signal subspaces)",
    )
    parser.add_argument(
        "--out_npy",
        metavar="NPY_FILE",
        type=str,
        default=None,
        help="[output] numpy file to save MUSIC spectrogram (time,direction=> power)",
    )
    parser.add_argument(
        "--out_full_npy",
        metavar="NPY_FILE",
        type=str,
        default=None,
        help="[output] numpy file to save MUSIC spectrogram (time,frequency,direction=> power",
    )
    parser.add_argument(
        "--out_fig",
        metavar="FIG_FILE",
        type=str,
        default=None,
        help="[output] fig file to save MUSIC spectrogram (.png)",
    )
    parser.add_argument(
        "--out_csv",
        metavar="CSV",
        type=str,
        default=None,
        help="[output] csv file to save MUSIC spectrogram (.csv)",
    )
    parser.add_argument(
        "--out_fig_with_bar",
        metavar="FIG_FILE",
        type=str,
        default=None,
        help="[output] fig file to save MUSIC spectrogram with color bar(.png)",
    )
    parser.add_argument(
        "--out_spectrogram_fig",
        metavar="FIG_FILE",
        type=str,
        default=None,
        help="[output] fig file to save power spectrogram (first channel) (.png)",
    )
    parser.add_argument(
        "--out_spectrogram_csv",
        metavar="csv",
        type=str,
        default=None,
        help="[output] fig file to save power spectrogram (first channel) (.csv)",
    )
    parser.add_argument(
        "--out_setting",
        metavar="SETTING_FILE",
        type=str,
        default=None,
        help="[output] stting file (.json)",
    )
    ####
    parser.add_argument(
        "--thresh",
        metavar="F",
        type=float,
        default=None,
        help="threshold of MUSIC power spectrogram",
    )
    parser.add_argument(
        "--event_min_size",
        metavar="W",
        type=int,
        default=3,
        help="minimum event size (MUSIC frame)",
    )
    parser.add_argument(
        "--out_localization",
        metavar="LOC_FILE",
        type=str,
        default=None,
        help="[output] localization file(.json)",
    )

    args = parser.parse_args()
    if not args:
        quit()
    #
    # read tf
    print("... reading", args.tf_filename)
    tf_config = read_hark_tf(args.tf_filename)

    # print positions of microphones
    # mic_pos=read_hark_tf_param(args.tf_filename)
    # print "# mic positions:",mic_pos

    spec, m_power, m_full_power, setting = music.compute_music_power(
        args.wav_filename,
        tf_config,
        args.normalize_factor,
        args.stft_win_size,
        args.stft_step,
        args.min_freq,
        args.max_freq,
        args.music_src_num,
        args.music_win_size,
        args.music_step,
    )

    # save setting
    if args.out_setting:
        outfilename = args.out_setting
        fp = open(outfilename, "w")
        json.dump(setting, fp, sort_keys=True, indent=2)
        print("[save]", outfilename)
    # save MUSIC spectrogram
    if args.out_npy:
        outfilename = args.out_npy
        np.save(outfilename, m_power)
        print("[save]", outfilename)
    # save MUSIC spectrogram for each freq.
    if args.out_full_npy:
        outfilename = args.out_full_npy
        np.save(outfilename, m_full_power)
        print("[save]", outfilename)
    # plot heat map
    if args.out_fig:
        music.save_heatmap_music_spec(args.out_fig, m_power)
    if args.out_csv:
        print("[save]", args.out_csv)
        with open(args.out_csv, "w") as fp:
            for i in range(len(m_power)):
                line=",".join(map(str,m_power[i,:]))
                fp.write(line)
                fp.write("\n")
    # plot heat map with color bar
    if args.out_fig_with_bar:
        music.save_heatmap_music_spec_with_bar(args.out_fig_with_bar, m_power)
    # plot spectrogram
    if args.out_spectrogram_fig:
        music.save_spectrogram(args.out_spectrogram_fig, spec, ch=0)
    if args.out_spectrogram_csv:
        print("[save]", args.out_spectrogram_csv)
        ch=0
        with open(args.out_spectrogram_csv, "w") as fp:
            for i in range(len(spec[ch])):
                v=np.absolute(spec[ch,i,:])
                line=",".join(map(str,v))
                fp.write(line)
                fp.write("\n")
    ####
    #### Detection part
    ####

    # threshold
    threshold = args.thresh

    def threshold_filter(x):
        if x < threshold:
            return 0
        else:
            return x

    f = np.vectorize(threshold_filter)
    m_power_thresh = f(m_power)
    print("# MUSIC power after thresholding")
    print(m_power_thresh)
    # peak detection
    peak_tl = []
    for i in range(m_power_thresh.shape[0]):
        # m_power_thresh[i]
        peaks = detect_peak(m_power_thresh[i])
        peak_tl.append(peaks)

    # tracking
    n_dir = m_power.shape[1]

    def diff_peak(a, b):
        d = abs(a - b)
        if d < n_dir / 2:
            return d
        else:
            return n_dir - d

    if len(peak_tl) == 1:
        print("[ERROR] too short", file=sys.stderr)
    # detect next frame
    tracking_peaks = [[] for t in range(len(peak_tl))]
    for t in range(len(peak_tl) - 1):
        p1s = peak_tl[t]
        p2s = peak_tl[t + 1]
        # search combination of peaks
        for i1, p1 in enumerate(p1s):
            nearest_p2 = None
            for i2, p2 in enumerate(p2s):
                if nearest_p2 == None or diff_peak(p1, p2) < nearest_p2[1]:
                    if diff_peak(p1, p2) < 6:
                        nearest_p2 = (i2, p2, abs(p1 - p2))
            tracking_peaks[t].append([i1, p1, nearest_p2, None])
    # for last event
    ps = peak_tl[len(peak_tl) - 1]
    for i1, p1 in enumerate(ps):
        tracking_peaks[len(peak_tl) - 1].append([i1, p1, None, None])
    # make id
    print("# detected peaks and IDs")
    id_cnt = 0
    for t in range(len(peak_tl) - 1):
        evts = tracking_peaks[t]
        evts2 = tracking_peaks[t + 1]
        for evt in evts:
            p2 = evt[2]
            # index,dir,next,count
            if evt[3] == None:
                evt[3] = id_cnt
                id_cnt += 1
            print(">>", evt)
            if p2 is not None:
                i2 = p2[0]
                # index,dir,diff
                evts2[i2][3] = evt[3]
    # count id
    id_counter = {}
    for t in range(len(peak_tl)):
        evts = tracking_peaks[t]
        for evt in evts:
            if evt[3] is not None:
                if not evt[3] in id_counter:
                    id_counter[evt[3]] = 0
                id_counter[evt[3]] += 1
    print("# event counting( ID => count ):", id_counter)
    # cut off & re-set id
    event_min_size = args.event_min_size
    print("# cut off: minimum event size:", event_min_size)
    reset_id_counter = {}
    tl_objs = []
    for t in range(len(peak_tl)):
        evts = tracking_peaks[t]
        objs = []
        for evt in evts:
            if evt[3] is not None and id_counter[evt[3]] >= event_min_size:
                poss = tf_config["positions"]
                pos = poss[evt[1]]
                eid = evt[3]
                if not eid in reset_id_counter:
                    reset_id_counter[eid] = len(reset_id_counter)
                new_eid = reset_id_counter[eid]
                obj = {"id": new_eid, "x": [pos[1], pos[2], pos[3]], "power": 1}
                objs.append(obj)
        tl_objs.append(objs)

    print("# re-assigned IDs")
    for o in tl_objs:
        print(o)
    save_obj = {"interval": setting["music_step_ms"], "tl": tl_objs}
    if args.out_localization is not None:
        filename = args.out_localization
        print("[save]", filename)
        with open(filename, "w") as f:
            json.dump(save_obj, f)

if __name__ == "__main__":
    main()
