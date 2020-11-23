```sh
python ensmic/preprocessing/prepare_fs.xray.py
```

```sh
nohup sh -c "PYTHONUNBUFFERED=x python ensmic/phase_i/training.py -m 'x-ray' --gpu 2" &> log.phase_i.x-ray.training.txt &
```
