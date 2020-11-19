# An analysis on Ensemble Learning optimized Neural Network Classification for COVID-19 CT and X-Ray Imaging

A survey on...?

ensmic?

nohup sh -c "PYTHONUNBUFFERED=x python phase_one/training.py 2>&1 | tee results/phase_one.x-ray/output.log" &

nohup 2>&1 sh -c "./run.sh" &> log.txt &

nohup sh -c "python covidxscan/validation/validate_screening.py --architecture VGG16 --gpu 0 >output.VGG16.log 2>&1" &


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


https://towardsdatascience.com/ensembles-the-almost-free-lunch-in-machine-learning-91af7ebe5090


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

### Phase 4:
Utilizing data augmentation for inference.

One model -> Multiple predictions on data augmentated testing -> ensemble learning -> final prediction

1x train-model & 1x val-model
-> Multiple predictions on 1x val-ensemble & 1x test set
-> Ensemble Learning on 1x val-ensemble
-> Evaluation on 1x test set

--------------------------------------------------------------------------------

## Usage

```sh
python ensmic/preprocessing/prepare_fs.xray.py
```

```sh
python ensmic/phase_one/training.py -m "x-ray"
python ensmic/phase_one/training.py -m "ct"
```

--------------------------------------------------------------------------------
Interesting datasets:
- https://www.kaggle.com/plameneduardo/sarscov2-ctscan-dataset?select=non-COVID
- https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/
