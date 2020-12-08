# An analysis on Ensemble Learning optimized Medical Image Classification with Deep Convolutional Neural Networks

Project Description etc

https://towardsdatascience.com/ensembles-the-almost-free-lunch-in-machine-learning-91af7ebe5090

--------------------------------------------

ensmeble learning output
class & probability/confidence (for ROC)

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

Additional:  
memory vs performance  
model complexity vs performance  

#### First Results

![PhaseI_F1](docs/plot.F1.png)

![PhaseI_FDR](docs/plot.FDR.png)

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

## Reproducibility

**Requirements:**
- Ubuntu 18.04
- Python 3.6
- NVIDIA QUADRO RTX 6000 or a GPU with equivalent performance

**Step-by-Step workflow:**

asd



## Datasets

#### X-Ray COVID19

**Classes:** 3 - Pneumonia, COVID-19, NORMAL  
**Size:** 2905 images  
**Source:** https://www.kaggle.com/tawsifurrahman/covid19-radiography-database  

**Short Description:**  
A team of researchers from Qatar University, Doha, Qatar and the University of Dhaka, Bangladesh along with their collaborators from Pakistan and Malaysia in collaboration with medical doctors have created a database of chest X-ray images for COVID-19 positive cases along with Normal and Viral Pneumonia images. In our current release, there are 219 COVID-19 positive images, 1341 normal images and 1345 viral pneumonia images. We will continue to update this database as soon as we have new x-ray images for COVID-19 pneumonia patients.

**Reference:**  
M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam, “Can AI help in screening Viral and COVID-19 pneumonia?” IEEE Access, Vol. 8, 2020, pp. 132665 - 132676.

#### The ISIC 2019 Challenge Dataset

**Classes:** 9 - Melanoma, Melanocytic nevus, Basal cell carcinoma, Actinic keratosis, Benign keratosis, Dermatofibroma, Vascular lesion, Squamous cell carcinoma, Unknown  
**Size:** 25,331 images  
**Source:** https://challenge2019.isic-archive.com/ or https://www.kaggle.com/andrewmvd/isic-2019

**Short Description:**  
Skin cancer is the most common cancer globally, with melanoma being the most deadly form. Dermoscopy is a skin imaging modality that has demonstrated improvement for diagnosis of skin cancer compared to unaided visual inspection. However, clinicians should receive adequate training for those improvements to be realized. In order to make expertise more widely available, the International Skin Imaging Collaboration (ISIC) has developed the ISIC Archive, an international repository of dermoscopic images, for both the purposes of clinical training, and for supporting technical research toward automated algorithmic analysis by hosting the ISIC Challenges.

**Note:**  
We didn't use the newest ISIC 2020 (https://challenge2020.isic-archive.com/), because it was purely a binary classification dataset.  
We utilized the multi-class 2019 variant in order to obtain a more difficult task for better evaluation of the ensemble learning performance gain.  

**Reference:**  
[1] Tschandl P., Rosendahl C. & Kittler H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi.10.1038/sdata.2018.161 (2018)  
[2] Noel C. F. Codella, David Gutman, M. Emre Celebi, Brian Helba, Michael A. Marchetti, Stephen W. Dusza, Aadi Kalloo, Konstantinos Liopyris, Nabin Mishra, Harald Kittler, Allan Halpern: “Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the 2017 International Symposium on Biomedical Imaging (ISBI), Hosted by the International Skin Imaging Collaboration (ISIC)”, 2017; arXiv:1710.05006.  
[3] Marc Combalia, Noel C. F. Codella, Veronica Rotemberg, Brian Helba, Veronica Vilaplana, Ofer Reiter, Allan C. Halpern, Susana Puig, Josep Malvehy: “BCN20000: Dermoscopic Lesions in the Wild”, 2019; arXiv:1908.02288.  


#### Retinal Image Analysis for multi-Disease Detection Challenge

**Classes:** 2 - Normal, Abnormal  
**Size:** 1920 images  
**Source:** https://riadd.grand-challenge.org/Home/  

**Short Description:**  
According to the WHO,  World report on vision 2019, the number of visually impaired people worldwide is estimated to be 2.2 billion, of whom at least 1 billion have a vision impairment that could have been prevented or is yet to be addressed. The world faces considerable challenges in terms of eye care, including inequalities in the coverage and quality of prevention, treatment, and rehabilitation services. Early detection and diagnosis of ocular pathologies would enable forestall of visual impairment.

**Reference:**  
https://riadd.grand-challenge.org/Home/


-> Replace with DDSM?

## Author

Dominik Müller\
Email: dominik.mueller@informatik.uni-augsburg.de\
IT-Infrastructure for Translational Medical Research\
University Augsburg\
Bavaria, Germany

## How to cite / More information

Coming soon

```
Coming soon
```

Thank you for citing our work.

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE Version 3.\
See the LICENSE.md file for license rights and limitations.
