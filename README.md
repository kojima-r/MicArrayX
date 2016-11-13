# SimMch

# セットアップ
```
$ ./setup.sh
```

# 簡単な使い方
```
./make_sim_wav.sh ./i1.wav ./i2.wav ./o1.wav ./o2.wav
```
上のコマンドで
./i1.wav ./i2.wav
から混合音をシミュレーションで作成し
再分離した結果を
./o1.wav ./o2.wav
に保存される。
（デフォルトでは０度方向と９０度方向に音源がある場合のシミュレーションを行う）

# make_sim_wav.sh の中で使われている各種スクリプト
- sim_tf.py
一つの音ファイルに伝達関数をかける
- wav_mix.py
複数の音ファイルを混合する
- const_sep.py
HARKを呼び出して、固定方向からの音を分離する


# MUSIC 
```
python music.py  ./sample/tamago_rectf.zip  ./sample/jinsei_tanosii.wav --out_npy test.npy --out_full_n
py test.npy --out_fig ./music.png --out_fig_with_bar ./music_bar.png --out_spectrogram ./fft.png 
```


# そのほかのスクリプトファイル
- simmch.py
本プロジェクトで使われているユーティリティ関数
- make_noise.py
ホワイトノイズの音ファイルを作成する
- sim_td.py
時間差から多チャンネル音を合成する（伝達関数を使わない合成）
- check_parseval.py
パーセバルの等式が成り立っていることを確認する。（FFTのチェック用）


