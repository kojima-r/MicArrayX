from hark_tf.read_param import read_hark_tf_param
from hark_tf.read_mat import read_hark_tf
import sys
import wave
import math
import numpy as np
import numpy.random as npr
import array
from scipy import hamming, interpolate, linalg

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt
import matplotlib.cm as cm


import micarrayx
from micarrayx.simulator.sim_tf import *
import micarrayx.localization.music
from micarrayx.filter.gsc import *

import json

def compute_music_power(
    spec,
    fs,
    tf_config,
    normalize_factor,
    fftLen,
    stft_step,
    min_freq,
    max_freq,
    src_num,
    music_win_size,
    music_step,
):
    setting = {}
    # reading data
    df = fs * 1.0 / fftLen
    # cutoff bin
    min_freq_bin = int(np.ceil(min_freq / df))
    max_freq_bin = int(np.floor(max_freq / df))
    print("# min freq. :", min_freq_bin * df, "Hz")
    setting["min_freq"] = min_freq_bin * df
    print("# max freq. :", max_freq_bin * df, "Hz")
    setting["max_freq"] = max_freq_bin * df
    print("# freq. step:", df, "Hz")
    setting["freq_step"] = df
    print("# min freq. bin index:", min_freq_bin)
    print("# max freq. bin index:", max_freq_bin)

    # apply STFT
    """
    win = np.hamming(fftLen)  # ハミング窓
    print(len(win))
    print(stft_step)
    spec = micarrayx.stft_mch(wav, win, stft_step)
    """
    spec_m = spec[:, :, min_freq_bin:max_freq_bin]
    # apply MUSIC method
    ## power[frame, freq, direction_id]
    print("# src_num:", src_num)
    setting["src_num"] = src_num
    print("# src_num:", src_num)
    setting["src_num"] = src_num
    setting["step_ms"] = 1000.0 / fs * stft_step
    setting["music_step_ms"] = 1000.0 / fs * stft_step * music_step
    power = micarrayx.localization.music.compute_music_spec(
        spec_m,
        src_num,
        tf_config,
        df,
        min_freq_bin,
        win_size=music_win_size,
        step=music_step,
    )
    p = np.sum(np.real(power), axis=1)
    m_power = 10 * np.log10(p + 1.0)
    m_full_power = 10 * np.log10(np.real(power) + 1.0)
    return spec, m_power, m_full_power, setting


import argparse
def main():
    # argv check
    parser = argparse.ArgumentParser(
        description="applying the MUSIC method to am-ch wave file"
    )
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
        "--out_recons",
        metavar="FILE",
        type=str,
        default="recons.wav",
        help="",
    )
    parser.add_argument(
        "--direction",
        metavar="V",
        type=float,
        default=45.0,
        help="",
    )
    parser.add_argument(
        "--distance",
        metavar="V",
        type=float,
        default=10.0,
        help="",
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
        "--out_fig_with_bar",
        metavar="FIG_FILE",
        type=str,
        default=None,
        help="[output] fig file to save MUSIC spectrogram with color bar(.png)",
    )
    parser.add_argument(
        "--out_spectrogram",
        metavar="FIG_FILE",
        type=str,
        default=None,
        help="[output] fig file to save power spectrogram (first channel) (.png)",
    )
    parser.add_argument(
        "--out_setting",
        metavar="SETTING_FILE",
        type=str,
        default=None,
        help="[output] stting file (.json)",
    )

    args = parser.parse_args()
    if not args:
        quit()
    # argv check
    
    npr.seed(1234)
    mic_pos = read_hark_tf_param(args.tf_filename)
    tf_config = read_hark_tf(args.tf_filename)
    print("# mic positions:", mic_pos)
    wav_filename = args.wav_filename
    wr = wave.open(wav_filename, "rb")
    src_theta = args.direction * math.pi / 180.0
    src_distance = args.distance
    src_index = micarrayx.nearest_direction_index(tf_config, src_theta)
    
    a_vec = get_beam_vec(tf_config, src_index)
    print("# mic positions  :", mic_pos)
    print("# direction index:", src_index)
    if not src_index in tf_config["tf"]:
        print(
            "Error: tf index", src_index, "does not exist in TF file", file=sys.stderr
        )
        quit()

    ## read wav file
    print("... reading", wav_filename)
    wav_data = micarrayx.read_mch_wave(wav_filename)
    scale = 32767.0
    wav = wav_data["wav"] / scale
    fs = wav_data["framerate"]
    nch = wav_data["nchannels"]
    ## print info
    print("# channel num : ", nch)
    print("# sample size : ", wav.shape)
    print("# sampling rate : ", fs)
    print("# sec : ", wav_data["duration"])
    mono_wavdata = wav[0, :]

    ## apply TF
    fftLen = 512
    step = fftLen / 4
    """
    mch_wavdata = apply_tf(mono_wavdata, fftLen, step, tf_config, src_index)
    a = np.max(mch_wavdata)
    mch_wavdata = mch_wavdata / a
    """
    mch_wavdata_spec = apply_tf_spec(mono_wavdata, fftLen, step, tf_config, src_index)
    
    spec, m_power, m_full_power, setting = compute_music_power(
        mch_wavdata_spec,
        fs,
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
        micarrayx.localization.music.save_heatmap_music_spec(args.out_fig, m_power)
    # plot heat map with color bar
    if args.out_fig_with_bar:
        micarrayx.localization.music.save_heatmap_music_spec_with_bar(args.out_fig_with_bar, m_power)
    # plot spectrogram
    if args.out_spectrogram:
        micarrayx.localization.music.save_spectrogram(args.out_spectrogram, spec, ch=0)   
    
    spec1 = mch_wavdata_spec[:, :, : int(fftLen / 2 + 1)]
    print("[beam forming input]>>",spec1.shape)
    #spec1 = micarrayx.stft_mch(wav1, win, step)
    # spec1[ch, frame, freq_bin]
    nch = spec1.shape[0]
    nframe = spec1.shape[1]
    nfreq_bin = spec1.shape[2]
    ### DS beamformer & blocked signals
    ds_freq = np.zeros((nframe, nfreq_bin), dtype=complex)
    for t in range(spec1.shape[1]):
        for freq_bin in range(spec1.shape[2]):
            ds_freq[t, freq_bin] = (
                np.dot(a_vec.conj()[freq_bin, :], spec1[:, t, freq_bin]) / nch
            )
    ds_freq = np.array([ds_freq])
    win = np.hamming(fftLen)  # ハミング窓
    step= args.stft_step
    recons_ds = micarrayx.istft_mch(ds_freq, win, step)
    micarrayx.save_mch_wave(recons_ds * 32767.0, args.out_recons)
    #

if __name__ == "__main__":
    main()
