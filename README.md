# covid-xscan
Detection of COVID-19 infection on x-rays via deep learning classification


RSNA: https://www.kaggle.com/c/10338/download-all


```sh
python covidxscan/validation/prepare_screening.py

nohup python covidxscan/validation/validate_screening.py --architecture VGG16 --gpu 0 2>&1 | tee output.VGG16.log &
nohup python covidxscan/validation/validate_screening.py --architecture Xception --gpu 2 2>&1 | tee output.Xception.log &
nohup python covidxscan/validation/validate_screening.py --architecture DenseNet --gpu 3 2>&1 | tee output.DenseNet.log &

nohup python covidxscan/validation/validate_screening.py --architecture ResNeSt --gpu 0 2>&1 | tee output.ResNeSt.log &
nohup python covidxscan/validation/validate_screening.py --architecture InceptionResNetV2 --gpu 1 2>&1 | tee output.InceptionResNetV2.log &

python covidxscan/validation/evaluate_screening.py
```

Starting Time: 10:04 (07.09.2020) - ServerTime 08:04

VGG16: 17230MiB
InceptionResNetV2: 17411MiB
Xception: 18211MiB
DenseNet: 11011MiB





--------------------------------------------

Pre-Sampling:
- 65% train-model
- 10% val-model
- 10% val-ensemble
- 15% test

## Analysis Phases:

### Phase 1:

Analyze all architectures with 1x train-model & 1x val-model   
->   Predict on 1x val-ensemble and 1x test set

### Phase 2:

Analyze all implemented ensemble techniques given all predictions of 1) and validation set -> Testing  
(can model importance be calculated?)  

Train each ensemble method on models predictions for 1x val-ensemble set

Run classifier on predictions for 1x test set

### Phase 3:
Use top-3 models and top-3 ensemble methods:  

Setup:  
for each top3 architecture:
  model-set = train-model + val-model
  perform 5-CV on model-set
  for each fold:
    predict on val-ensemble
    predict on test
  for each top3 ensembler:
    train ensembler on 5-CV models predictions for val-ensemble
    predict on test using the 5-CV model predictions


Note: 3-CV instead of 5-CV?

--------------------------------------------------------------------------------

Interesting datasets:
- https://www.kaggle.com/plameneduardo/sarscov2-ctscan-dataset?select=non-COVID
- https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/
