tf=./sample/tamago_rectf.zip
wav=./sample/jinsei.wav
ch=0 # target channel of input wav files
dir=0 # degree

# generating sound
python ./sim_tf.py ${tf} ${wav} ${ch} ${dir} 1 ./sample/jinsei_recons.wav

