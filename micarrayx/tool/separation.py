from hark_tf.read_param import read_hark_tf_param
from hark_tf.read_mat import read_hark_tf
import sys
import wave
import math
import numpy as np
import numpy.random as npr
import array
from scipy import hamming, interpolate, linalg

import micarrayx
from micarrayx.simulator.sim_tf import *
import micarrayx.localization.music
from micarrayx.filter.gsc import *

import json
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
    
    ### separation setting
    parser.add_argument(
        "--timeline",
        type=str,
        default="tl.json",
        help="",
    )
    
    ### stft 
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
    ### output 
    parser.add_argument(
        "--out",
        metavar="FILE",
        type=str,
        default="sep",
        help="[output] prefix of separated output wav files",
    )
    parser.add_argument(
        "--out_sep_spectrogram_fig",
        action="store_true",
    )
    parser.add_argument(
        "--out_sep_spectrogram_csv",
        action="store_true",
    )

    ## argv check
    args = parser.parse_args()
    if not args:
        quit()
    
    npr.seed(1234)
    ## read tf file
    mic_pos = read_hark_tf_param(args.tf_filename)
    tf_config = read_hark_tf(args.tf_filename)
    print("# mic positions:", mic_pos)
   
    ## read wav file
    wav_filename = args.wav_filename
    print("... reading", wav_filename)
    wav_data = micarrayx.read_mch_wave(wav_filename)
    scale = 32767.0
    wav = wav_data["wav"] / scale
    fs = wav_data["framerate"]
    nch = wav_data["nchannels"]
    print("# channel num : ", nch)
    print("# sample size : ", wav.shape)
    print("# sampling rate : ", fs)
    print("# sec : ", wav_data["duration"])

    ## apply STFT
    fftLen = args.stft_win_size#512
    win = np.hamming(fftLen)  # ハミング窓
    spec = micarrayx.stft_mch(wav, win, args.stft_step)
    time_step=args.stft_step*1000.0/fs

    ## read timeline file
    timeline_data = json.load(open(args.timeline))
    interval=timeline_data["interval"]
    tl=timeline_data["tl"]

    ### DS beamformer & blocked signals
    # spec[ch, frame, freq_bin]
    print("[beam forming input]>>",spec.shape)
    nch = spec.shape[0]
    nframe = spec.shape[1]
    nfreq_bin = spec.shape[2]
    ds_freq = np.zeros((nframe, nfreq_bin), dtype=complex)
    sep_specs={}
    for t in range(nframe):
        current_time=t*time_step
        current_idx=int(current_time/interval)
        #print(t,current_idx)
        if current_idx < len(tl):
            events=tl[current_idx]
            for e in events:
                theta = math.atan2(e["x"][1],e["x"][0])
                index = micarrayx.nearest_direction_index(tf_config, theta)
                a_vec = get_beam_vec(tf_config, index)
                ds_freq = np.zeros((nfreq_bin,), dtype=complex)
                for freq_bin in range(nfreq_bin):
                    ds_freq[freq_bin] = (
                        np.dot(a_vec.conj()[freq_bin, :], spec[:, t, freq_bin]) / nch
                    )
                eid=e["id"]
                if eid not in sep_specs:
                    sep_specs[eid]=[]
                sep_specs[eid].append(ds_freq)
    ## save separated wav files
    for eid, sep_spec in sep_specs.items():
        #print(eid)
        ds_freq=np.array([sep_spec])
        recons_ds = micarrayx.istft_mch(ds_freq, win, args.stft_step)
        ### save files
        out_filename=args.out+"."+str(eid)+".wav"
        print("[SAVE]",out_filename)
        micarrayx.save_mch_wave(recons_ds * 32767.0, out_filename)
        if args.out_sep_spectrogram_fig:
            out_filename=args.out+"."+str(eid)+".spec.png"
            micarrayx.localization.music.save_spectrogram(out_filename, ds_freq, ch=0)
        if args.out_sep_spectrogram_csv:
            out_filename=args.out+"."+str(eid)+".spec.csv"
            print("[SAVE]",out_filename)
            ch=0
            with open(out_filename, "w") as fp:
                for i in range(len(ds_freq[ch])):
                    v=np.absolute(ds_freq[ch,i,:])
                    line=",".join(map(str,v))
                    fp.write(line)
                    fp.write("\n")

if __name__ == "__main__":
    main()
