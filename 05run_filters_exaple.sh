micarrayx-make-noise sample/noise.wav --template ./sample/jinsei_tanosii_recons.wav
micarrayx-mix ./sample/jinsei_tanosii_recons.wav ./sample/noise.wav -o ./sample/jinsei_noise_mix.wav -N 1,1


micarrayx-filter-wiener ./sample/jinsei_noise_mix.wav ./sample/noise.wav --noise aaa
micarrayx-filter-wiener ./sample/jinsei_noise_mix.wav ./sample/jinsei.wav --tf ./sample/tamago_rectf.zip
micarrayx-filter-gsc    ./sample/tamago_rectf.zip    ./sample/jinsei_noise_mix.wav
