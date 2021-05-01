## Setup File Structure

```sh
nohup sh -c "python ensmic/data_loading/prepare_fs.covid.py" &> log.prepare_fs.covid.txt &
nohup sh -c "python ensmic/data_loading/prepare_fs.isic.py" &> log.prepare_fs.isic.txt &
nohup sh -c "python ensmic/data_loading/prepare_fs.chmnist.py" &> log.prepare_fs.chmnist.txt &
nohup sh -c "python ensmic/data_loading/prepare_fs.drd.py" &> log.prepare_fs.drd.txt &
```

# Phase Baseline

```sh
# Dataset: COVID-19
nohup sh -c "python ensmic/phase_baseline/training.py -m 'covid' --gpu 1" &> log.phase_baseline.covid.training.txt &
nohup sh -c "python ensmic/phase_baseline/inference.py -m 'covid' --gpu 3" &> log.phase_baseline.covid.inference.txt &

# Dataset: CHMNIST
nohup sh -c "python ensmic/phase_baseline/training.py -m 'chmnist' --gpu 3" &> log.phase_baseline.chmnist.training.txt &
nohup sh -c "python ensmic/phase_baseline/inference.py -m 'chmnist' --gpu 1" &> log.phase_baseline.chmnist.inference.txt &

# Dataset: Diabetic Retinopathy Detection
nohup sh -c "python ensmic/phase_baseline/training.py -m 'drd' --gpu 0" &> log.phase_baseline.drd.training.txt &
nohup sh -c "python ensmic/phase_baseline/inference.py -m 'drd' --gpu 3" &> log.phase_baseline.drd.inference.txt &

# Dataset: ISIC 2019
nohup sh -c "python ensmic/phase_baseline/training.py -m 'isic' --gpu 0" &> log.phase_baseline.isic.training.txt &
nohup sh -c "python ensmic/phase_baseline/inference.py -m 'isic' --gpu 3" &> log.phase_baseline.isic.inference.txt &
```

# Phase Augmenting

```sh
# Dataset: COVID-19
nohup sh -c "python ensmic/phase_augmenting/inference.py -m 'covid' --gpu 3" &> log.phase_augmenting.covid.inference.txt &
# Dataset: CHMNIST
nohup sh -c "python ensmic/phase_augmenting/inference.py -m 'chmnist' --gpu 1" &> log.phase_augmenting.chmnist.inference.txt &
# Dataset: ISIC 2019
nohup sh -c "python ensmic/phase_augmenting/inference.py -m 'isic' --gpu 3" &> log.phase_augmenting.isic.inference.txt &
```
