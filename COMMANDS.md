```sh
python ensmic/preprocessing/prepare_fs.covid.py
python ensmic/preprocessing/prepare_fs.isic.py
```

```sh
nohup sh -c "python ensmic/phase_i/training.py -m 'covid' --gpu 0" &> log.phase_i.covid.training.txt &
```
