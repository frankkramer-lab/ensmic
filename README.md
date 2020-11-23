# An analysis on Ensemble Learning optimized Neural Network Classification for COVID-19 CT and X-Ray Imaging

A survey on...?

ensmic?


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
Utilizing data augmentation for inference.

One model -> Multiple predictions on data augmentated testing -> ensemble learning -> final prediction

1x train-model & 1x val-model
-> Multiple predictions on 1x val-ensemble & 1x test set
-> Ensemble Learning on 1x val-ensemble
-> Evaluation on 1x test set

### Phase 4:
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

## First Results

**Phase I:**  

![PhaseI_F1](docs/plot.F1.png)

![PhaseI_FDR](docs/plot.FDR.png)

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
