## Setup File Structure

```sh
python ensmic/preprocessing/prepare_fs.covid.py
python ensmic/preprocessing/prepare_fs.isic.py
python ensmic/preprocessing/prepare_fs.riadd.py
```

# Phase I Analysis

```sh
# Dataset: COVID-19
nohup sh -c "python ensmic/phase_i/training.py -m 'covid' --gpu 0" &> log.phase_i.covid.training.txt &

# Dataset: ISIC 2019
nohup sh -c "python ensmic/phase_i/training.py -m 'isic' --gpu 1" &> log.phase_i.isic.training.txt &

# Dataset: RIADD 2021
nohup sh -c "python ensmic/phase_i/training.py -m 'riadd' --gpu 2" &> log.phase_i.riadd.training.txt &
```
