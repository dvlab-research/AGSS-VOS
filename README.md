# AGSS-VOS: Attention Guided Single-Shot Video Object Segmentation

## Prerequisites

- Python 3.6
- NVIDIA GPU
- Ubuntu
- Pytorch 0.4.0

## Model training and evaluation

### Data Preparation
1. Download [YouTubeVOS](https://drive.google.com/open?id=1bI5J1H3mxsIGo7Kp-pPZU8i6rnykOw7f).
2. Download [DAVIS-2017](https://davischallenge.org/davis2017/code.html).
3. We prepare the split annotations for all dataset and meta.json for davis2017 in [here](https://drive.google.com/drive/folders/1sSYjIbuPieL3XfM4lFR0THEjGZyGM2qq?usp=sharing)
4. Symlink the corresponding train/validation/test dataset and json files to `data` folder. 
```
data
├── youtube_vos
│   ├── train
│   │   ├── JPEGImages
│   │   ├── Split_Annotations
│   │   ├── meta.json
│   ├── valid
│   │   ├── JPEGImages
│   │   ├── Split_Annotations
│   │   ├── meta.json
│   ├── valid_all_frames
│   │   ├── JPEGImages
├── davis2017
│   ├── trainval
│   │   ├── JPEGImages
│   │   ├── Split_Annotations
│   │   ├── train_meta.json
│   │   ├── val_meta.json
│   ├── test
│   │   ├── JPEGImages
│   │   ├── Split_Annotations
│   │   ├── test_meta.json
```

### Model Preparation
1. Download [RGMP](https://www.dropbox.com/s/gt0kivrb2hlavi2/weights.pth?dl=0) and place it (weigths.pth) in the 'checkpoints' folder.
2. Download [the pretrained model](https://drive.google.com/drive/folders/1sSYjIbuPieL3XfM4lFR0THEjGZyGM2qq?usp=sharing) and place it in the 'checkpoints' folder. 
```
checkpoints
├── weights.pth
├── train_ytv
│   ├── model_4.pth
├── train_davis
│   ├── model_199.pth
├── ft_davis
│   ├── model_99.pth
```
3. Download [FlowNet2C](https://drive.google.com/file/d/1BFT6b7KgKJC8rA59RmOVAXRM_S7aSfKE/view?usp=sharing) and place it in the 'flow_inference/models/FlowNet2-C_checkpoint.pth.tar'. You need to run
```
sh mask.sh
```
in the 'channelnorm_package', 'correlation_package' and 'resample2d_package' in 'flow_inference/networks_flow/' folder. Make sure the version of PyTorch is '0.4.0'.

### Training
1. To train on Youtube-VOS training set.
``` 
sh run_ytv.sh
```
2. To train on DAVIS-2017 training set.
``` 
sh run_davis.sh
```
The checkpoint will be saved in the 'Outputs' folder.

### Finetuning
1. To finetune on the DAVIS-2017 training set with pretrained model on Youtube-VOS traininset.
```
sh ft_davis.sh
```

### Validation
1. To inference on Youtube-VOS validation set.
```
sh val_ytv.sh
```
2. To inference on DAVIS-2017 validation set.
```
sh val_davis.sh
```
3. To inference on DAVIS-2017 test-dev set.
```
sh test_davis.sh
```
4. To inference on DAVIS-2017 validation/test-dev set with finetuned model.
```
sh val_davis_ft.sh
```
```
sh test_davis_ft.sh
```

The results will be saved in the 'val_dir' or 'test_dir' folder.
You can change the '--restore' in scripts to validate your own training result.


#### This software is for Non-commercial Research Purposes only.

### Contact
If you have any questions, please feel free to contact the authors.

Huaijia Lin <linhj@cse.cuhk.edu.hk> or <huaijialin@gmail.com>

### Citation

If you use our code, please consider citing our paper:

```
@inproceedings{lin2019agss,
  title={AGSS-VOS: Attention Guided Single-Shot Video Object Segmentation},
  author={Lin, Huaijia and Qi, Xiaojuan and Jia, Jiaya},
  booktitle={ICCV},
  year={2019}
}
```

## Acknowledgments
Parts of this code were derived from [RGMP](https://github.com/xanderchf/RGMP) and [FlowNet2](https://github.com/NVIDIA/flownet2-pytorch).

