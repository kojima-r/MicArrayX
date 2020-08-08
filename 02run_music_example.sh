tf=./sample/tamago_rectf.zip
wav=./sample/jinsei.wav

micarrayx-localize-music ${tf} ${wav} --out_npy test.npy --out_full_npy test_full.npy --out_fig ./music.png --out_fig_with_bar ./music_bar.png --out_spectrogram ./spectrogram.png --out_setting ./music.json

