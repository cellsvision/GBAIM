# GBAIM

## Environment  
Tested on **Ubuntu 16.04**, Python **3.7**  

## Requirements  
Python packages required:  
- `torch`==1.10.0+cu111
- `numpy`==1.19.5
- `opencv-python`==4.5.3.56 
- `scikit-learn`==0.24.2  

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