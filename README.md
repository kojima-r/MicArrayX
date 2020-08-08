# MicArrayX

A signal processing library for multi-channel sound recorded by a microphone array.
- Simulation of microphone array recording: 
a multi-channel recording is simulated using a single-channel sound source file (wav file) and a transfer function or geometric calculation.
- Localoization and separation: this library also implements a basic algorithm for sound source localization and sound source separation.

# Installation
```
pip install git+https://github.com/kojima-r/MicArrayX.git
```

# Example

## Initilization

The following command downloads the data such as transfer functions and prepares to carry out the examples.
```
$ sh 00init_example.sh
```

## Simulation

The first example simulates the microphone array recording
where
- sound source: ./sample/jinsei.wav (0th channel is used)
- transfer function: ../sample/tamago_rectf.zip which simulates an 8-channel microphone array
- location of the sound source: zero-degree direction, i.e., in front of a microphone array.
This recorded sound is saved into ./sample/jinsei_recons.wav by the following commands:
```
01run_sim_example.sh
```

``` 01run_sim_example.sh
tf=./sample/tamago_rectf.zip
wav=./sample/jinsei_tanosii.wav
ch=0 # target channel of input wav files
dir=0 # degree

# generating sound
micarrayx-sim-tf ${tf} ${wav} ${ch} ${dir} 1 ./sample/jinsei_tanosii_recons.wav
```

In this scripts, `micarrayx-sim-tf` command is used.

1. Tranfer function file (HARK2.0 format)
2. Sound source wav file
3. Which channel of the input sound file is used
 - Just one channel can be used as a given sound source wav file
4. Scale value
 - Usually, 1 is specified. If you want to change the volume, please specify 1 or less. If the transfer function reduces the sound, please　adjust this option by 1 or more.
5. Output sound file.
 - Note that it will be overwritten.

## Localization

MUSIC (MUltiple SIgnal Classification) is a standard sound souce localization method.
The following command carried the MUSIC method:
```
$ 02run_music_example.sh
```
This file contain the following scripts:
```
tf=./sample/tamago_rectf.zip
wav=./sample/jinsei.wav

micarrayx-localize-music ${tf} ${wav} 
  --out_npy test.npy
  --out_full_npy test_full.npy
  --out_fig ./music.png
  --out_fig_with_bar ./music_bar.png
  --out_spectrogram ./spectrogram.png
  --out_setting ./music.json
```

1. Tranfer function file (HARK2.0 format) for a microphone array.
2. Sound source wav file recorded by the microphone array 1.

The remaining options are related to output files.
3. `--out_npy`: MUSIC spectorgram
4. `--out_full_npy`: MUSIC spectorgram for each frequency bin
5. `--out_fig`: heat-map plot of the MUSIC spectorgram
6. `--out_fig_with_bar`: heat-map plot with a color bar of the MUSIC spectorgram
7. `--out_spectrogram`: spectrogram
8. `--out_setting`: setting of input sound source and MUSIC methods

`micarrayx-localize-music` command only computes MUSIC spectrogram, i.e., the power of sounds for each direction.
To detect sound souces and localize them, the following commands is used and `03run_music_example.sh`  is reffered.

```
micarrayx-localize ./sample/tamago_rectf.zip ./sample/jinsei.wav --thresh 33
```

`micarrayx-localize` perform the MUSIC method and thresholding with `--thresh` option.
This command detects the peak of MUSIC spectrogram and output the direction of sound souces.

## Separation

```
sh 05run_filters_exaple.sh
```
This file contain the following scripts:

```
micarrayx-make-noise sample/noise.wav --template ./sample/jinsei_tanosii_recons.wav
micarrayx-mix ./sample/jinsei_tanosii_recons.wav ./sample/noise.wav -o ./sample/jinsei_noise_mix.wav -N 1,1


micarrayx-filter-wiener ./sample/jinsei_noise_mix.wav ./sample/noise.wav --noise aaa
micarrayx-filter-wiener ./sample/jinsei_noise_mix.wav ./sample/jinsei.wav --tf ./sample/tamago_rectf.zip
micarrayx-filter-gsc    ./sample/tamago_rectf.zip    ./sample/jinsei_noise_mix.wav
```

The first two lines creates a noisy recording file:
`micarrayx-make-noise` outputs the white noise into `sample/noise.wav` with the same duration and channel as `./sample/jinsei_tanosii_recons.wav`
The`micarrayx-mix`
command outputs  `./sample/jinsei_noise_mix.wav`, a mixture of `./sample/jinsei_tanosii_recons.wav` and  `./sample/noise.wav ` at a ratio 1:1.

`micarrayx-filter-wiener` and  `micarrayx-filter-gsc` carries out a wiener filter and GSC filter, respectively.

## Utility commands and scripts

- `micarrayx-make-noise` creates a white noise sound file
- `micarrayx-sim-td` synthesizes a multi-channel sound from time difference without transfer function.
- `micarrayx/check_parseval.py` makes sure the Parseval equation holds (for checking FFT).

##　Creating simulation data（under construction） 
```
./make_sim_wav.sh ./i1.wav ./i2.wav ./o1.wav ./o2.wav
```
This command simulates mixture sounds from 
`./i1.wav` and  `./i2.wav`, and 
outputs the re-separated sounds into 
`./o1.wav` and `./o2.wav`.
By default, simulation is performed such that there exists sound sources in the 0° and 90° directions)

In this scripts,
`example_hark/const_sep.py` launches HARK and separates the mixture sound from specified directions.


