# -*- coding: utf-8 -*-
""" command: micarrayx-sim-td
"""
from hark_tf.read_param import read_hark_tf_param
import sys
import wave
import math
import numpy as np
import numpy.random as npr
import array


def gen_random_source(num, near, far, num_type):
    ret = {}
    for index in range(num):
        rate = npr.rand()
        d = near + (far - near) * rate
        rate_rad = npr.rand() * 2 - 1
        theta = math.pi * rate_rad
        type_id = npr.randint(num_type)
        ret[index] = {"distance": d, "azimuth": theta, "elevation": 0, "type": type_id}
    return ret


def gen_const_source(d, theta, type_id, num=1):
    ret = {}
    for index in range(num):
        ret[index] = {"distance": d, "azimuth": theta, "elevation": 0, "type": type_id}
    return ret


def rec_source(wavdata, src_wavdata, start_frame):
    end_frame = src_wavdata.shape[0] + start_frame
    print("start:", start_frame)
    print("end  :", end_frame)
    if wavdata.shape[0] < end_frame:
        diff_frame = end_frame - wavdata.shape[0]
        wavdata = np.r_[wavdata, np.zeros(diff_frame, dtype=wavdata.dtype)]
    wavdata[start_frame:end_frame] = src_wavdata
    return wavdata


def concat_waves(waves):
    max_len = max([w.shape[0] for w in waves])
    aligned_waves = [
        np.r_[w, np.zeros(max_len - w.shape[0], dtype="int16")] for w in waves
    ]
    return np.vstack(aligned_waves)


def p2o(d, theta, phi):
    z = d * math.sin(phi)
    pd = d * math.cos(phi)
    x = pd * math.cos(theta)
    y = pd * math.sin(theta)
    return np.array([x, y, z])


def compute_delay(mic_positions, src_info, t_c):
    v = 20.055 * ((t_c + 273.15) ** (1 / 2.0))
    print("# speed of sound:", v)
    res = {}
    for pos in mic_positions:
        index = pos[0]
        x = pos[1]
        y = pos[2]
        z = pos[3]
        mic_pos = np.array([x, y, z])
        src_d = src_info["distance"]
        src_theta = src_info["azimuth"]
        src_phi = src_info["elevation"]
        src_pos = p2o(src_d, src_theta, src_phi)
        d = np.sqrt(np.sum((mic_pos - src_pos) ** 2))
        res[index] = d / v
    return res


def compute_volume(mic_positions, src_info, near, near_v):
    c = near * near_v
    res = {}
    for pos in mic_positions:
        index = pos[0]
        x = pos[1]
        y = pos[2]
        z = pos[3]
        mic_pos = np.array([x, y, z])
        src_d = src_info["distance"]
        src_theta = src_info["azimuth"]
        src_phi = src_info["elevation"]
        src_pos = p2o(src_d, src_theta, src_phi)
        d = np.sqrt(np.sum((mic_pos - src_pos) ** 2))
        res[index] = c / d
    return res


def delaytime2sample(mic_delay, samplingrate):
    res = {}
    for key, val in list(mic_delay.items()):
        res[key] = int(val * samplingrate)
    return res


def main():
    # argv check
    if len(sys.argv) < 4:
        print(
            "Usage: sim_td.py <in: tf.zip(HARK2 transfer function file)> <in: src.wav> <out: dest.wav> [in: distance] [in: theta(degree]",
            file=sys.stderr,
        )
        quit()
    #
    npr.seed(1234)
    tf_filename = sys.argv[1]
    mic_pos = read_hark_tf_param(tf_filename)
    print("# mic positions:", mic_pos)
    wav_filename = sys.argv[2]
    wr = wave.open(wav_filename, "rb")
    out_filename = sys.argv[3]
    src_theta = None
    src_distance = None
    if len(sys.argv) >= 6:
        src_theta = float(sys.argv[4]) * math.pi / 180.0
        src_distance = float(sys.argv[5])

    # print info
    print("# channel num : ", wr.getnchannels())
    print("# sample size : ", wr.getsampwidth())
    print("# sampling rate : ", wr.getframerate())
    print("# frame num : ", wr.getnframes())
    print("# params : ", wr.getparams())
    print("# sec : ", float(wr.getnframes()) / wr.getframerate())

    # reading data
    data = wr.readframes(wr.getnframes())
    nch = wr.getnchannels()
    wavdata = np.frombuffer(data, dtype="int16")
    samplingrate = wr.getframerate()
    mono_wavdata = wavdata[1::nch]
    wr.close()

    # gen source
    src_num = 1
    src_near = 1
    src_far = 10
    src_type = 1
    srcs = None
    if src_distance != None and src_theta != None:
        srcs = gen_random_source(src_num, src_near, src_far, src_type)
    else:
        srcs = gen_const_source(src_distance, src_theta, 0)
    print("# src :", srcs)

    # compute time delay
    mic_delay = compute_delay(mic_pos, srcs[0], 0)
    mic_volume = compute_volume(mic_pos, srcs[0], src_near, 0.9)
    mic_delay_sample = delaytime2sample(mic_delay, samplingrate)
    print("# delay (sec):", mic_delay)
    print("# delay (sample):", mic_delay_sample)

    # shift samples with time delay
    all_wav = [np.zeros(0, dtype="int16") for i in range(len(mic_pos))]
    all_delay_wav = [
        rec_source(w, mono_wavdata, mic_delay_sample[index]) * mic_volume[index]
        for index, w in enumerate(all_wav)
    ]
    print(all_delay_wav)

    # concatnate mics(channels)
    mch_wavdata = concat_waves(all_delay_wav).transpose()
    print(mch_wavdata.shape)
    # save data
    out_wavdata = mch_wavdata.copy(order="C")
    print(out_wavdata.shape)
    ww = wave.Wave_write(out_filename)
    ww.setparams(wr.getparams())
    ww.setnchannels(out_wavdata.shape[1])
    ww.setnframes(out_wavdata.shape[0])
    ww.writeframes(array.array("h", out_wavdata.astype("int16").ravel()).tobytes())
    ww.close()

if __name__ == "__main__":
    main()
