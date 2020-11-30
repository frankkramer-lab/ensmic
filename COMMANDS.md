```sh
python ensmic/preprocessing/prepare_fs.xray.py
```

```sh
nohup sh -c "python ensmic/phase_i/training.py -m 'x-ray' --gpu 0" &> log.phase_i.x-ray.training.txt &
```
