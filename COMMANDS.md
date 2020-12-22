## Setup File Structure

```sh
nohup sh -c "python ensmic/preprocessing/prepare_fs.covid.py" &> log.prepare_fs.covid.txt &
nohup sh -c "python ensmic/preprocessing/prepare_fs.isic.py" &> log.prepare_fs.isic.txt &
nohup sh -c "python ensmic/preprocessing/prepare_fs.chmnist.py" &> log.prepare_fs.chmnist.txt &
nohup sh -c "python ensmic/preprocessing/prepare_fs.drd.py" &> log.prepare_fs.drd.txt &
```

# Phase I Analysis

```sh
# Dataset: COVID-19
nohup sh -c "python ensmic/phase_i/training.py -m 'covid' --gpu 0" &> log.phase_i.covid.training.txt &
nohup sh -c "python ensmic/phase_i/inference.py -m 'covid' --gpu 0" &> log.phase_i.covid.inference.txt &

# Dataset: CHMNIST
nohup sh -c "python ensmic/phase_i/training.py -m 'chmnist' --gpu 0" &> log.phase_i.chmnist.training.txt &
nohup sh -c "python ensmic/phase_i/inference.py -m 'chmnist' --gpu 2" &> log.phase_i.chmnist.inference.txt &

# Dataset: Diabetic Retinopathy Detection
nohup sh -c "python ensmic/phase_i/training.py -m 'drd' --gpu 3" &> log.phase_i.drd.training.txt &
nohup sh -c "python ensmic/phase_i/inference.py -m 'drd' --gpu 3" &> log.phase_i.drd.inference.txt &

# Dataset: ISIC 2019
nohup sh -c "python ensmic/phase_i/training.py -m 'isic' --gpu 0" &> log.phase_i.isic.training.txt &
nohup sh -c "python ensmic/phase_i/inference.py -m 'isic' --gpu 2" &> log.phase_i.isic.inference.txt &
```
