
micarrayx-sim-tf ./sample/microcone_geotf.zip ./sample/jinsei.wav 0 90 1 ./sample/test_geotf00.wav > sample/result_geotf.txt
micarrayx-sim-tf ./sample/microcone_rectf.zip ./sample/jinsei.wav 0 90 1 ./sample/test_rectf00.wav > sample/result_rectf.txt
micarrayx-sim-td ./sample/microcone_geotf.zip ./sample/jinsei.wav ./sample/test_td00.wav 10 90 > sample/result_td.txt

micarrayx-sim-tf ./sample/microcone_geotf.zip ./sample/jinsei.wav 1 90 1 ./sample/test_geotf00.wav 
micarrayx-make-noise sample/noise01.wav --template ./sample/test_geotf00.wav
micarrayx-mix ./sample/noise01.wav ./sample/test_geotf00.wav -o mixed.wav

python example_hark/const_sep.py  -i example_hark/const_sep.n.tmpl -t ./sample/microcone_geotf.zip -d 90,0 mixed.wav 

