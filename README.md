# GBAIM
This repository contains the code for the paper **"A multi-class gastric biopsy artificial intelligence model (GBAIM) developed from whole slide histopathological images: a retrospective and prospective study"**.  

## Abstract  

Background and aim: Pathological diagnosis of biopsies is the gold standard for identifying gastric diseases, particularly gastric carcinoma, but it suffers from clinical misdiagnoses and heavy workload. Furthermore, accurately distinguishing between early-stage gastric carcinoma (EGC) and advanced-stage gastric carcinoma (AGC) during diagnosis is essential for precise treatment; however, it remains a challenge in biopsies. In this study, we aim to develop an artificial intelligence (AI) model to address these challenges.  
Methods: Data from one internal and five external medical centers, spanning 2017 to 2022, were retrospectively collected, comprising 20,711 whole-slide images (WSIs) from 17,086 patients. In 2023-2024, a prospective cohort of 3,698 WSIs from 2,965 patients was additionally enrolled. Gastric Biopsy AI Model (GBAIM) were developed to perform six-class classification on WSIs and further fine-tuned to differentiate EGC from AGC. Besides, an auxiliary diagnostic experiment was conducted with 300 WSIs and nine pathologists who were grouped as junior, intermediate and senior pathologists.   
Results: GBAIM achieved 96.1% sensitivity and 95.0% specificity in external cohorts, and 93.4% and 99.0%, respectively, in the prospective cohort, demonstrating excellent performance and strong generalization ability. The fine-tuned GBAIM-T achieved an AUC of 0.907 in EGC/AGC classification on the internal testing set, outperforming intermediate and senior pathologists, and achieved an AUC of 0.826 on the independent external testing set. GBAIM enhanced the performance of all nine pathologists in auxiliary diagnostic experiment, raised their accuracy by 1.7%-39.0%, while reducing diagnostic time by 29.4%-50.5%.   
Conclusions: GBAIM is proved to be a valuable AI tool in clinical practice. Moreover, GBAIM is, to our best knowledge, the first model capable of GC-staging in gastric biopsies. 
## Environment  
Tested on **Ubuntu 16.04**, Python **3.7**  

## Requirements  
Python packages required:  
- `torch`==1.10.0+cu111
- `numpy`==1.19.5
- `opencv-python`==4.5.3.56 
- `scikit-learn`==0.24.2
- `timm`==0.6.12  

Install dependencies with:  
```bash
pip install -r requirements.txt
```


## Patch Level Model Training 
* Training dataset CSV: ./data/datasets/sample_patch_train_data.csv
* Validation dataset CSV: ./data/datasets/sample_patch_val_data.csv
* Patch images directory: ./data/patches/
  
The directory structure for patches should be:
```bash
│── WSI_001/
│   ├── patch_001.png
│   ├── patch_002.png
│   ├── ...
│── WSI_002/
│   ├── patch_001.png
│   ├── patch_002.png
│   ├── ...
```

To start patch-level model training, run the following command:
```bash
python ./train_val.py --do_train
```

To continue patch-level model training from a previous checkpoint, run the following command:
```bash 
python ./train_val.py --do_train --ckpt_path [checkpoint_path]
```

To start patch-level model evaluation, run the following command:
```bash
python ./train_val.py --ckpt_path [checkpoint_path]
```

## WSI-Level Inference
After training, sliding window inference will be perform on the WSIs, and the features will be saved in pickle format, following the structure in ./data/pkl 

To generate the WSI-level result:
```bash 
python ./get_wsi_result.py --ckpt_path [checkpoint_path]
```
