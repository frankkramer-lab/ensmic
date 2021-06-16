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
nohup sh -c "python ensmic/phase_baseline/evaluation.py -m 'covid'" &> log.phase_baseline.covid.evaluation.txt &

# Dataset: CHMNIST
nohup sh -c "python ensmic/phase_baseline/training.py -m 'chmnist' --gpu 3" &> log.phase_baseline.chmnist.training.txt &
nohup sh -c "python ensmic/phase_baseline/inference.py -m 'chmnist' --gpu 1" &> log.phase_baseline.chmnist.inference.txt &
nohup sh -c "python ensmic/phase_baseline/evaluation.py -m 'chmnist'" &> log.phase_baseline.chmnist.evaluation.txt &

# Dataset: Diabetic Retinopathy Detection
nohup sh -c "python ensmic/phase_baseline/training.py -m 'drd' --gpu 0" &> log.phase_baseline.drd.training.txt &
nohup sh -c "python ensmic/phase_baseline/inference.py -m 'drd' --gpu 0" &> log.phase_baseline.drd.inference.txt &
nohup sh -c "python ensmic/phase_baseline/evaluation.py -m 'drd'" &> log.phase_baseline.drd.evaluation.txt &

# Dataset: ISIC 2019
nohup sh -c "python ensmic/phase_baseline/training.py -m 'isic' --gpu 0" &> log.phase_baseline.isic.training.txt &
nohup sh -c "python ensmic/phase_baseline/inference.py -m 'isic' --gpu 3" &> log.phase_baseline.isic.inference.txt &
nohup sh -c "python ensmic/phase_baseline/evaluation.py -m 'isic'" &> log.phase_baseline.isic.evaluation.txt &
```

# Phase Augmenting

```sh
# Dataset: COVID-19
nohup sh -c "python ensmic/phase_augmenting/inference.py -m 'covid' --gpu 3" &> log.phase_augmenting.covid.inference.txt &
nohup sh -c "python ensmic/phase_augmenting/evaluation.py -m 'covid'" &> log.phase_augmenting.covid.evaluation.txt &

# Dataset: CHMNIST
nohup sh -c "python ensmic/phase_augmenting/inference.py -m 'chmnist' --gpu 1" &> log.phase_augmenting.chmnist.inference.txt &
nohup sh -c "python ensmic/phase_augmenting/evaluation.py -m 'chmnist'" &> log.phase_augmenting.chmnist.evaluation.txt &

# Dataset: Diabetic Retinopathy Detection
nohup sh -c "python ensmic/phase_augmenting/inference.py -m 'drd' --gpu 0" &> log.phase_augmenting.drd.inference.txt &
nohup sh -c "python ensmic/phase_augmenting/evaluation.py -m 'drd'" &> log.phase_augmenting.drd.evaluation.txt &

# Dataset: ISIC 2019
nohup sh -c "python ensmic/phase_augmenting/inference.py -m 'isic' --gpu 3" &> log.phase_augmenting.isic.inference.txt &
nohup sh -c "python ensmic/phase_augmenting/evaluation.py -m 'isic'" &> log.phase_augmenting.isic.evaluation.txt &
```

# Phase Stacking

```sh
# Dataset: COVID-19
nohup sh -c "python ensmic/phase_stacking/prepare.py -m 'covid'" &> log.phase_stacking.covid.prepare.txt &
nohup sh -c "python ensmic/phase_stacking/train_inf.py -m 'covid'" &> log.phase_stacking.covid.train_inf.txt &
nohup sh -c "python ensmic/phase_stacking/evaluation.py -m 'covid'" &> log.phase_stacking.covid.evaluation.txt &

# Dataset: CHMNIST
nohup sh -c "python ensmic/phase_stacking/prepare.py -m 'chmnist'" &> log.phase_stacking.chmnist.prepare.txt &
nohup sh -c "python ensmic/phase_stacking/train_inf.py -m 'chmnist'" &> log.phase_stacking.chmnist.train_inf.txt &
nohup sh -c "python ensmic/phase_stacking/evaluation.py -m 'chmnist'" &> log.phase_stacking.chmnist.evaluation.txt &

# Dataset: Diabetic Retinopathy Detection
nohup sh -c "python ensmic/phase_stacking/prepare.py -m 'drd'" &> log.phase_stacking.drd.prepare.txt &
nohup sh -c "python ensmic/phase_stacking/train_inf.py -m 'drd'" &> log.phase_stacking.drd.train_inf.txt &
nohup sh -c "python ensmic/phase_stacking/evaluation.py -m 'drd'" &> log.phase_stacking.drd.evaluation.txt &

# Dataset: ISIC 2019
nohup sh -c "python ensmic/phase_stacking/prepare.py -m 'isic'" &> log.phase_stacking.isic.prepare.txt &
nohup sh -c "python ensmic/phase_stacking/train_inf.py -m 'isic'" &> log.phase_stacking.isic.train_inf.txt &
nohup sh -c "python ensmic/phase_stacking/evaluation.py -m 'isic'" &> log.phase_stacking.isic.evaluation.txt &
```

# Phase Bagging

```sh
# Dataset: COVID-19 - Setup
nohup sh -c "python ensmic/phase_bagging/prepare.py -m 'covid'" &> log.phase_bagging.covid.prepare.txt &
# Dataset: COVID-19 - Training
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'covid' -a 'Vanilla' --gpu 0" &> log.phase_bagging.covid.model_train.Vanilla.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'covid' -a 'EfficientNetB4' --gpu 0" &> log.phase_bagging.covid.model_train.EfficientNetB4.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'covid' -a 'ResNet101' --gpu 1" &> log.phase_bagging.covid.model_train.ResNet101.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'covid' -a 'ResNeXt101' --gpu 1" &> log.phase_bagging.covid.model_train.ResNeXt101.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'covid' -a 'MobileNetV2' --gpu 1" &> log.phase_bagging.covid.model_train.MobileNetV2.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'covid' -a 'DenseNet121' --gpu 2" &> log.phase_bagging.covid.model_train.DenseNet121.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'covid' -a 'InceptionResNetV2' --gpu 2" &> log.phase_bagging.covid.model_train.InceptionResNetV2.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'covid' -a 'VGG16' --gpu 3" &> log.phase_bagging.covid.model_train.VGG16.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'covid' -a 'Xception' --gpu 3" &> log.phase_bagging.covid.model_train.Xception.txt &
# Dataset: COVID-19 - Inference
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'covid' -a 'Vanilla' --gpu 0" &> log.phase_bagging.covid.model_inf.Vanilla.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'covid' -a 'EfficientNetB4' --gpu 0" &> log.phase_bagging.covid.model_inf.EfficientNetB4.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'covid' -a 'ResNet101' --gpu 1" &> log.phase_bagging.covid.model_inf.ResNet101.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'covid' -a 'ResNeXt101' --gpu 1" &> log.phase_bagging.covid.model_inf.ResNeXt101.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'covid' -a 'MobileNetV2' --gpu 1" &> log.phase_bagging.covid.model_inf.MobileNetV2.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'covid' -a 'DenseNet121' --gpu 2" &> log.phase_bagging.covid.model_inf.DenseNet121.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'covid' -a 'InceptionResNetV2' --gpu 2" &> log.phase_bagging.covid.model_inf.InceptionResNetV2.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'covid' -a 'VGG16' --gpu 3" &> log.phase_bagging.covid.model_inf.VGG16.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'covid' -a 'Xception' --gpu 3" &> log.phase_bagging.covid.model_inf.Xception.txt &
# Dataset: COVID-19 - Ensemble
nohup sh -c "python ensmic/phase_bagging/ensemble_train_inf.py -m 'covid' --gpu 0" &> log.phase_bagging.covid.ensemble_train_inf.txt &
# Dataset: COVID-19 - Evaluation
nohup sh -c "python ensmic/phase_bagging/evaluation.py -m 'covid'" &> log.phase_bagging.covid.evaluation.txt &

#------------------------------------------------------------------------------#

# Dataset: CHMNIST - Setup
nohup sh -c "python ensmic/phase_bagging/prepare.py -m 'chmnist'" &> log.phase_bagging.chmnist.prepare.txt &
# Dataset: CHMNIST - Training
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'chmnist' -a 'Vanilla' --gpu 0" &> log.phase_bagging.chmnist.model_train.Vanilla.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'chmnist' -a 'EfficientNetB4' --gpu 0" &> log.phase_bagging.chmnist.model_train.EfficientNetB4.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'chmnist' -a 'ResNet101' --gpu 1" &> log.phase_bagging.chmnist.model_train.ResNet101.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'chmnist' -a 'ResNeXt101' --gpu 1" &> log.phase_bagging.chmnist.model_train.ResNeXt101.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'chmnist' -a 'MobileNetV2' --gpu 1" &> log.phase_bagging.chmnist.model_train.MobileNetV2.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'chmnist' -a 'DenseNet121' --gpu 2" &> log.phase_bagging.chmnist.model_train.DenseNet121.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'chmnist' -a 'InceptionResNetV2' --gpu 2" &> log.phase_bagging.chmnist.model_train.InceptionResNetV2.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'chmnist' -a 'VGG16' --gpu 3" &> log.phase_bagging.chmnist.model_train.VGG16.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'chmnist' -a 'Xception' --gpu 3" &> log.phase_bagging.chmnist.model_train.Xception.txt &


















# Dataset: CHMNIST - Inference
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'chmnist' -a 'Vanilla' --gpu 0" &> log.phase_bagging.chmnist.model_inf.Vanilla.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'chmnist' -a 'EfficientNetB4' --gpu 0" &> log.phase_bagging.chmnist.model_inf.EfficientNetB4.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'chmnist' -a 'ResNet101' --gpu 1" &> log.phase_bagging.chmnist.model_inf.ResNet101.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'chmnist' -a 'ResNeXt101' --gpu 1" &> log.phase_bagging.chmnist.model_inf.ResNeXt101.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'chmnist' -a 'MobileNetV2' --gpu 1" &> log.phase_bagging.chmnist.model_inf.MobileNetV2.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'chmnist' -a 'DenseNet121' --gpu 2" &> log.phase_bagging.chmnist.model_inf.DenseNet121.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'chmnist' -a 'InceptionResNetV2' --gpu 2" &> log.phase_bagging.chmnist.model_inf.InceptionResNetV2.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'chmnist' -a 'VGG16' --gpu 3" &> log.phase_bagging.chmnist.model_inf.VGG16.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'chmnist' -a 'Xception' --gpu 3" &> log.phase_bagging.chmnist.model_inf.Xception.txt &
# Dataset: CHMNIST - Ensemble
nohup sh -c "python ensmic/phase_bagging/ensemble_train_inf.py -m 'chmnist' --gpu 0" &> log.phase_bagging.chmnist.ensemble_train_inf.txt &
# Dataset: CHMNIST - Evaluation
nohup sh -c "python ensmic/phase_bagging/evaluation.py -m 'chmnist'" &> log.phase_bagging.chmnist.evaluation.txt &

#------------------------------------------------------------------------------#

# Dataset: Diabetic Retinopathy Detection - Setup
nohup sh -c "python ensmic/phase_bagging/prepare.py -m 'drd'" &> log.phase_bagging.drd.prepare.txt &
# Dataset: Diabetic Retinopathy Detection - Training
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'drd' -a 'Vanilla' --gpu 0" &> log.phase_bagging.drd.model_train.Vanilla.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'drd' -a 'EfficientNetB4' --gpu 0" &> log.phase_bagging.drd.model_train.EfficientNetB4.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'drd' -a 'ResNet101' --gpu 1" &> log.phase_bagging.drd.model_train.ResNet101.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'drd' -a 'ResNeXt101' --gpu 1" &> log.phase_bagging.drd.model_train.ResNeXt101.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'drd' -a 'MobileNetV2' --gpu 1" &> log.phase_bagging.drd.model_train.MobileNetV2.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'drd' -a 'DenseNet121' --gpu 2" &> log.phase_bagging.drd.model_train.DenseNet121.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'drd' -a 'InceptionResNetV2' --gpu 2" &> log.phase_bagging.drd.model_train.InceptionResNetV2.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'drd' -a 'VGG16' --gpu 3" &> log.phase_bagging.drd.model_train.VGG16.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'drd' -a 'Xception' --gpu 3" &> log.phase_bagging.drd.model_train.Xception.txt &
# Dataset: Diabetic Retinopathy Detection - Inference
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'drd' -a 'Vanilla' --gpu 0" &> log.phase_bagging.drd.model_inf.Vanilla.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'drd' -a 'EfficientNetB4' --gpu 0" &> log.phase_bagging.drd.model_inf.EfficientNetB4.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'drd' -a 'ResNet101' --gpu 1" &> log.phase_bagging.drd.model_inf.ResNet101.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'drd' -a 'ResNeXt101' --gpu 1" &> log.phase_bagging.drd.model_inf.ResNeXt101.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'drd' -a 'MobileNetV2' --gpu 1" &> log.phase_bagging.drd.model_inf.MobileNetV2.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'drd' -a 'DenseNet121' --gpu 2" &> log.phase_bagging.drd.model_inf.DenseNet121.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'drd' -a 'InceptionResNetV2' --gpu 2" &> log.phase_bagging.drd.model_inf.InceptionResNetV2.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'drd' -a 'VGG16' --gpu 3" &> log.phase_bagging.drd.model_inf.VGG16.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'drd' -a 'Xception' --gpu 3" &> log.phase_bagging.drd.model_inf.Xception.txt &
# Dataset: Diabetic Retinopathy Detection - Ensemble
nohup sh -c "python ensmic/phase_bagging/ensemble_train_inf.py -m 'drd' --gpu 0" &> log.phase_bagging.drd.ensemble_train_inf.txt &
# Dataset: Diabetic Retinopathy Detection - Evaluation
nohup sh -c "python ensmic/phase_bagging/evaluation.py -m 'drd'" &> log.phase_bagging.drd.evaluation.txt &

#------------------------------------------------------------------------------#

# Dataset: ISIC 2019 - Setup
nohup sh -c "python ensmic/phase_bagging/prepare.py -m 'isic'" &> log.phase_bagging.isic.prepare.txt &
# Dataset: ISIC 2019 - Training
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'isic' -a 'Vanilla' --gpu 0" &> log.phase_bagging.isic.model_train.Vanilla.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'isic' -a 'EfficientNetB4' --gpu 0" &> log.phase_bagging.isic.model_train.EfficientNetB4.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'isic' -a 'ResNet101' --gpu 1" &> log.phase_bagging.isic.model_train.ResNet101.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'isic' -a 'ResNeXt101' --gpu 1" &> log.phase_bagging.isic.model_train.ResNeXt101.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'isic' -a 'MobileNetV2' --gpu 1" &> log.phase_bagging.isic.model_train.MobileNetV2.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'isic' -a 'DenseNet121' --gpu 2" &> log.phase_bagging.isic.model_train.DenseNet121.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'isic' -a 'InceptionResNetV2' --gpu 2" &> log.phase_bagging.isic.model_train.InceptionResNetV2.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'isic' -a 'VGG16' --gpu 3" &> log.phase_bagging.isic.model_train.VGG16.txt &
nohup sh -c "python ensmic/phase_bagging/model_train.py -m 'isic' -a 'Xception' --gpu 3" &> log.phase_bagging.isic.model_train.Xception.txt &
# Dataset: ISIC 2019 - Inference
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'isic' -a 'Vanilla' --gpu 0" &> log.phase_bagging.isic.model_inf.Vanilla.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'isic' -a 'EfficientNetB4' --gpu 0" &> log.phase_bagging.isic.model_inf.EfficientNetB4.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'isic' -a 'ResNet101' --gpu 1" &> log.phase_bagging.isic.model_inf.ResNet101.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'isic' -a 'ResNeXt101' --gpu 1" &> log.phase_bagging.isic.model_inf.ResNeXt101.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'isic' -a 'MobileNetV2' --gpu 1" &> log.phase_bagging.isic.model_inf.MobileNetV2.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'isic' -a 'DenseNet121' --gpu 2" &> log.phase_bagging.isic.model_inf.DenseNet121.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'isic' -a 'InceptionResNetV2' --gpu 2" &> log.phase_bagging.isic.model_inf.InceptionResNetV2.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'isic' -a 'VGG16' --gpu 3" &> log.phase_bagging.isic.model_inf.VGG16.txt &
nohup sh -c "python ensmic/phase_bagging/model_inf.py -m 'isic' -a 'Xception' --gpu 3" &> log.phase_bagging.isic.model_inf.Xception.txt &
# Dataset: ISIC 2019 - Ensemble
nohup sh -c "python ensmic/phase_bagging/ensemble_train_inf.py -m 'isic' --gpu 0" &> log.phase_bagging.isic.ensemble_train_inf.txt &
# Dataset: ISIC 2019 - Evaluation
nohup sh -c "python ensmic/phase_bagging/evaluation.py -m 'isic'" &> log.phase_bagging.isic.evaluation.txt &

```
