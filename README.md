# covid-xscan
Detection of COVID-19 infection on x-rays via deep learning classification


RSNA: https://www.kaggle.com/c/10338/download-all


```sh
python covidxscan/validation/prepare_screening.py

python covidxscan/validation/validate_screening.py --architecture VGG16 --gpu 0 2>&1 | tee output.VGG16.log &
python covidxscan/validation/validate_screening.py --architecture InceptionResNetV2 --gpu 1 | tee output.InceptionResNetV2.log &
python covidxscan/validation/validate_screening.py --architecture Xception --gpu 2 | tee output.Xception.log &
python covidxscan/validation/validate_screening.py --architecture DenseNet --gpu 3 | tee output.DenseNet.log &
python covidxscan/validation/validate_screening.py --architecture ResNeSt --gpu 0 | tee output.ResNeSt.log &

python covidxscan/validation/evaluate_screening.py
```
