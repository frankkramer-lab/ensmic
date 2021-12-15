# An Analysis on Ensemble Learning optimized Medical Image Classification with Deep Convolutional Neural Networks

Novel and high-performance medical image classification pipelines are heavily utilizing ensemble learning strategies. The idea of ensemble learning is to assemble diverse models or multiple predictions and, thus, boost prediction performance. However, it is still an open question to what extend as well as which ensemble learning strategies are beneficial in deep learning based medical image classification pipelines.  

In this work, we proposed a reproducible medical image classification pipeline (ensmic) for analyzing the performance impact of the following ensemble learning techniques: Augmenting, Stacking, and Bagging. The pipeline consists of state-of-the-art preprocessing and image augmentation methods as well as 9 deep convolution neural network architectures. It was applied on four popular medical imaging datasets with varying complexity. Furthermore, 12 pooling functions for combining multiple predictions were analyzed, ranging from simple statistical functions like unweighted averaging up to more complex learning-based functions like support vector machines.  

![Schema](docs/schema.png)

We concluded that the integration of Stacking and Augmentation ensemble learning techniques is a powerful method for any medical image classification pipeline to improve robustness and boost performance.

The sampling, results, figures and meta data is available under the following link:  
Coming soon

--------------------------------------------

## Results

![Showcase](docs/figure.showcase.jpg)

Our results revealed that Stacking was able to achieve the largest performance gain of up to 13% F1-score increase. Augmenting showed consistent improvement capabilities by up to 4% and is also appliable to single model based pipelines. Cross-validation based Bagging demonstrated to be the most complex ensemble learning method, which resulted in an F1-score decrease in all analyzed datasets (up to -10%). Furthermore, we demonstrated that simple statistical pooling functions are equal or often even better than more complex pooling functions.

![Results](docs/figure.comparison.final.png)

Summary of all experiments to identify performance impact of ensemble learning techniques on medical image classification.  
LEFT: Bar plots showing the maximum achieved Accuracy across all methods for each ensemble learning technique and dataset: Baseline (red), Augmenting (blue), Bagging (green) and Stacking (purple). Additionally, the distribution of achieved F1-scores by the various methods is illustrated with box plots.  
RIGHT: Computed performance impact between the best scoring method of the Baseline and the best scoring method of the applied ensemble learning technique for each dataset. The performance impact is represented as performance gain in % between F1-scores (RIGHT TOP) as well as Accuracies (RIGHT BOTTOM). The color mapping of the ensemble learning techniques are equal to Figure 7 LEFT (Augmenting: Blue; Bagging: Green; Stacking: Purple).

## Reproducibility

**Requirements:**
- Ubuntu 18.04
- Python 3.7
- NVIDIA QUADRO RTX 6000 or a GPU with equivalent performance

**Step-by-Step workflow:**

Download ensmic via:
```sh
git clone https://github.com/frankkramer-lab/ensmic.git
cd ensmic/
```

Install ensmic via:
```sh
python setup.py install
```

Run the scripts for the desired phases.  
Please check out the following protocol on script execution: https://github.com/frankkramer-lab/ensmic/blob/master/COMMANDS.md

---------------------------

## Datasets

#### X-Ray COVID19

**Classes:** 3 - Pneumonia, COVID-19, NORMAL  
**Size:** 2.905 images  
**Source:** https://www.kaggle.com/tawsifurrahman/covid19-radiography-database  

**Short Description:**  
A team of researchers from Qatar University, Doha, Qatar and the University of Dhaka, Bangladesh along with their collaborators from Pakistan and Malaysia in collaboration with medical doctors have created a database of chest X-ray images for COVID-19 positive cases along with Normal and Viral Pneumonia images. In our current release, there are 219 COVID-19 positive images, 1341 normal images and 1345 viral pneumonia images. We will continue to update this database as soon as we have new x-ray images for COVID-19 pneumonia patients.

**Reference:**  
M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam, “Can AI help in screening Viral and COVID-19 pneumonia?” IEEE Access, Vol. 8, 2020, pp. 132665 - 132676.

#### The ISIC 2019 Challenge Dataset

**Classes:** 9 - Melanoma, Melanocytic nevus, Basal cell carcinoma, Actinic keratosis, Benign keratosis, Dermatofibroma, Vascular lesion, Squamous cell carcinoma, Unknown  
**Size:** 25.331 images  
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

#### Diabetic Retinopathy Detection Dataset

**Classes:** 5 - "No DR", "Mild", "Moderate", "Severe", "Proliferative DR"  
**Size:** 35.126 images  
**Source:** https://www.kaggle.com/c/diabetic-retinopathy-detection/overview  

**Short Description:**  
Diabetic retinopathy is the leading cause of blindness in the working-age population of the developed world. It is estimated to affect over 93 million people. Currently, detecting DR is a time-consuming and manual process that requires a trained clinician to examine and evaluate digital color fundus photographs of the retina. By the time human readers submit their reviews, often a day or two later, the delayed results lead to lost follow up, miscommunication, and delayed treatment. The need for a comprehensive and automated method of DR screening has long been recognized, and previous efforts have made good progress using image classification, pattern recognition, and machine learning. With color fundus photography as input, the goal of this competition is to push an automated detection system to the limit of what is possible – ideally resulting in models with realistic clinical potential. The winning models will be open sourced to maximize the impact such a model can have on improving DR detection.

**Reference:**  
https://www.kaggle.com/c/diabetic-retinopathy-detection/overview


#### Colorectal Histology MNIST

**Classes:** 8 - EMPTY, COMPLEX, MUCOSA, DEBRIS, ADIPOSE, STROMA, LYMPHO, TUMOR  
**Size:** 5.000 images  
**Source:** https://www.kaggle.com/kmader/colorectal-histology-mnist

**Short Description:**  
Automatic recognition of different tissue types in histological images is an essential part in the digital pathology toolbox. Texture analysis is commonly used to address this problem; mainly in the context of estimating the tumour/stroma ratio on histological samples. However, although histological images typically contain more than two tissue types, only few studies have addressed the multi-class problem. For colorectal cancer, one of the most prevalent tumour types, there are in fact no published results on multiclass texture separation. The dataset serves as a much more interesting MNIST or CIFAR10 problem for biologists by focusing on histology tiles from patients with colorectal cancer. In particular, the data has 8 different classes of tissue (but Cancer/Not Cancer can also be an interesting problem).

**Reference:**  
Kather JN, Weis CA, Bianconi F, Melchers SM, Schad LR, Gaiser T, Marx A, Zöllner FG. Multi-class texture analysis in colorectal cancer histology. Sci Rep. 2016 Jun 16;6:27988. doi: 10.1038/srep27988. PMID: 27306927; PMCID: PMC4910082.

---------------------------

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
