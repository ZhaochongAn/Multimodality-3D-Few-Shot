<p align="center">
  <h1 align="center">Multimodality Helps Few-shot 3D Point Cloud Semantic Segmentation</h1>
  <p align="center">
    <a href="https://zhaochongan.github.io/"><strong>Zhaochong An</strong></a>
    ·
    <a href="https://guoleisun.github.io/"><strong>Guolei Sun<sup>†</sup></strong></a>
    ·
    <a href="https://yun-liu.github.io/"><strong>Yun Liu<sup>†</sup></strong></a>
    ·
    <a href="https://runjiali-rl.github.io/"><strong>Runjia Li</strong></a>
    ·
    <a href="https://sites.google.com/site/wumincf/"><strong>Min Wu</strong></a>
    <br>
    <a href="https://mmcheng.net/cmm/"><strong>Ming-Ming Cheng</strong></a>
    ·
    <a href="https://people.ee.ethz.ch/~kender/"><strong>Ender Konukoglu</strong></a>
    ·
    <a href="https://sergebelongie.github.io/"><strong>Serge Belongie</strong></a>
  </p>
  <h2 align="center">ICLR 2025 Spotlight (<a href="https://arxiv.org/pdf/2410.22489">Paper</a>)</h2>
</p>

<p align="center">
  <img src="https://ZhaochongAn.github.io/images/MMFSS_github.png" alt="Overview" width="80%">
</p>

## 🌟 Highlights

We introduce:
- A novel **cost-free multimodal few-shot 3D point cloud segmentation (FS-PCS) setup** that integrates textual category names and 2D image modality
- **MM-FSS**: The first multimodal FS-PCS model that explicitly utilizes textual modality and implicitly leverages 2D modality
- Superior performance on novel class generalization through effective multimodal integration
- Valuable insights into the importance of commonly-ignored free modalities in FS-PCS

## 🛠️ Environment Setup

Our environment has been tested on:
- RTX 3090 GPUs
- GCC 6.3.0

Follow the [COSeg installation guide](https://github.com/ZhaochongAn/COSeg?tab=readme-ov-file#environment) for detailed setup.

## 📦 Dataset Preparation

### Pretraining Stage Data
Follow [OpenScene](https://github.com/pengsongyou/openscene?tab=readme-ov-file#data-preparation) instructions, you can 
directly download the following ScanNet 3D dataset and 2D features for pretraining:
```bash
# Download ScanNet 3D dataset
wget https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_3d.zip
unzip scannet_3d.zip

# Download 2D features
wget https://cvg-data.inf.ethz.ch/openscene/data/scannet_multiview_lseg.zip
unzip scannet_multiview_lseg.zip
```

You should put the unpacked data into the folder ./pretraining/data/ or link to the corresponding data folder with the symbolic link:
```bash
ln -s /PATH/TO/DOWNLOADED/FOLDER ./pretraining/data
```

### Few-shot Stage Data
#### Option 1: Direct Download (Recommended)
Download our preprocessed datasets:

| Dataset | Few-shot Stage Data |
|:-------:|:---------------------:|
| S3DIS | [Download](https://drive.google.com/file/d/1frJ8nf9XLK_fUBG4nrn8Hbslzn7914Ru/view?usp=drive_link) |
| ScanNet | [Download](https://drive.google.com/file/d/19yESBZumU-VAIPrBr8aYPaw7UqPia4qH/view?usp=drive_link) |

#### Option 2: Manual Preprocessing




Follow [COSeg](https://github.com/ZhaochongAn/COSeg?tab=readme-ov-file#datasets-preparation) preprocessing instructions.
The processed data will be in `[PATH_to_DATASET_processed_data]/blocks_bs1_s1/data`. Make sure to update the `data_root` entry in the .yaml 
config file to `[PATH_to_DATASET_processed_data]/blocks_bs1_s1/data`.


## 🔄 Training Pipeline

### 1. Backbone and IF Head Pretraining

**Option A**: Download our pretrained weights from [Google Drive](https://drive.google.com/drive/u/1/folders/1JoeAXJh1AZM3bM0KGBJQsFTad6uqpzUJ)

**Option B**: Train from scratch:
```bash
cd pretraining
bash run/distill_strat.sh PATH_to_SAVE_BACKBONE config/scannet/ours_lseg_strat.yaml
```

### 2. Meta-learning Stage
Set config `config/[CONFIG_FILE]` to be `s3dis_COSeg_fs.yaml` or `scannetv2_COSeg_fs.yaml` for training on S3DIS or ScanNet respectively.
Adjust `cvfold`, `n_way`, and `k_shot` according to your few-shot task:

```bash
# For 1-way tasks
python3 main_fs.py --config config/[CONFIG_FILE] \
    save_path [PATH_to_SAVE_MODEL] \
    pretrain_backbone [PATH_to_SAVED_BACKBONE] \
    cvfold [CVFOLD] \
    n_way 1 \
    k_shot [K_SHOT] \
    num_episode_per_comb 1000

# For 2-way tasks
python3 main_fs.py --config config/[CONFIG_FILE] \
    save_path [PATH_to_SAVE_MODEL] \
    pretrain_backbone [PATH_to_SAVED_BACKBONE] \
    cvfold [CVFOLD] \
    n_way 2 \
    k_shot [K_SHOT] \
    num_episode_per_comb 100
```

> **Note**: Following [COSeg](https://github.com/ZhaochongAn/COSeg?tab=readme-ov-file#training-pipeline), `num_episode_per_comb` defaults to 1000 for 1-way and 100 for 2-way tasks to maintain consistency in test set size.

## 📊 Evaluation & Visualization


### Model Evaluation
Modify `cvfold`, `n_way`, `k_shot` and `num_episode_per_comb` accordingly and run:
```bash
python3 main_fs.py --config config/[CONFIG_FILE] \
    test True \
    eval_split test \
    weight [PATH_to_SAVED_MODEL] \
    [vis 1]  # Optional: Enable W&B visualization
```

> **Note**: Performance may vary by 1.0% due to potential randomness in the training process. ScanNetv2 typically shows less variance than S3DIS.

### Visualization
Follow [COSeg visualization guide](https://github.com/ZhaochongAn/COSeg?tab=readme-ov-file#visualization) for high-quality visualization results.

## 🎯 Model Zoo

| Model | Dataset | CVFOLD | N-way K-shot | Weights |
|:-------:|:---------:|:--------:|:------------:|:----------:|
| s30_1w1s | S3DIS | 0 | 1-way 1-shot | [Download](https://drive.google.com/drive/u/1/folders/1XKxEnvT_VdVa9kP5P6DeXRQoC1YJyMK-) |
| s30_1w5s | S3DIS | 0 | 1-way 5-shot | [Download](https://drive.google.com/drive/u/1/folders/1dd3JmuLwLT6V03bsg_0J4ISLDnvoAUDq) |
| s30_2w1s | S3DIS | 0 | 2-way 1-shot | [Download](https://drive.google.com/drive/u/1/folders/1kJif7istSwHbsbeHQoI4sfQgDdF19T6v) |
| s30_2w5s | S3DIS | 0 | 2-way 5-shot | [Download](https://drive.google.com/drive/u/1/folders/1F17vApLTZFt2x85OjJtR6ryha0xDW6kV) |
| s31_1w1s | S3DIS | 1 | 1-way 1-shot | [Download](https://drive.google.com/drive/u/1/folders/1GK9pwWbti61mLxmCbSb40inr1QU42FgF) |
| s31_1w5s | S3DIS | 1 | 1-way 5-shot | [Download](https://drive.google.com/drive/u/1/folders/1EeyruLVk0ONXDBQ1W-pDVZAq6VPCC3jx) |
| s31_2w1s | S3DIS | 1 | 2-way 1-shot | [Download](https://drive.google.com/drive/u/1/folders/1m11yaTi7nm4_hfBWzZUAwM1G1I8kXZNj) |
| s31_2w5s | S3DIS | 1 | 2-way 5-shot | [Download](https://drive.google.com/drive/u/1/folders/1ytilaDjiHFUCqK-YGSqxDByWnWFFvvuR) |
| sc0_1w1s | ScanNet | 0 | 1-way 1-shot | [Download](https://drive.google.com/drive/u/1/folders/1krip2sLd9kkaq5viTdsoaPnFgRBG64w6) |
| sc0_1w5s | ScanNet | 0 | 1-way 5-shot | [Download](https://drive.google.com/drive/u/1/folders/1wGc3zv-ZwEpa_jNDSXWX64O4uRrYOFfI) |
| sc0_2w1s | ScanNet | 0 | 2-way 1-shot | [Download](https://drive.google.com/drive/u/1/folders/1rgLyb1Q6VoxgyQj_Eqfn4g-dcEeY-KYZ) |
| sc0_2w5s | ScanNet | 0 | 2-way 5-shot | [Download](https://drive.google.com/drive/u/1/folders/106_3fYBakpbMHwkknoEaGHFeIt42GAYW) |
| sc1_1w1s | ScanNet | 1 | 1-way 1-shot | [Download](https://drive.google.com/drive/u/1/folders/1fsljMc0lrqB-kMAQD85CmSU02qiFAt_z) |
| sc1_1w5s | ScanNet | 1 | 1-way 5-shot | [Download](https://drive.google.com/drive/u/1/folders/1MVEOV1ZZg3xQuwWhoNeeHpXBJ3kRCPEE) |
| sc1_2w1s | ScanNet | 1 | 2-way 1-shot | [Download](https://drive.google.com/drive/u/1/folders/1y_OVENsKy5RbeJ77CuwdJKXMbO_ZdtBx) |
| sc1_2w5s | ScanNet | 1 | 2-way 5-shot | [Download](https://drive.google.com/drive/u/1/folders/189HZgypuF9KWEVZ3tPW4bJ4QU1jk-2f-) |

## 📝 Citation
If you find our code or paper useful, please cite:


```bibtex
@article{an2025generalized,
  title={Generalized Few-shot 3D Point Cloud Segmentation with Vision-Language Model},
  author={An, Zhaochong and Sun, Guolei and Liu, Yun and Li, Runjia and Han, Junlin and Konukoglu, Ender and Belongie, Serge},
  journal={arXiv preprint arXiv:2503.16282},
  year={2025}
}

@article{an2024multimodality,
    title={Multimodality Helps Few-Shot 3D Point Cloud Semantic Segmentation},
    author={An, Zhaochong and Sun, Guolei and Liu, Yun and Li, Runjia and Wu, Min 
            and Cheng, Ming-Ming and Konukoglu, Ender and Belongie, Serge},
    journal={arXiv preprint arXiv:2410.22489},
    year={2024}
}

@inproceedings{an2024rethinking,
  title={Rethinking Few-shot 3D Point Cloud Semantic Segmentation},
  author={An, Zhaochong and Sun, Guolei and Liu, Yun and Liu, Fayao and Wu, Zongwei and Wang, Dan and Van Gool, Luc and Belongie, Serge},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3996--4006},
  year={2024}
}
```

For any questions or issues, feel free to reach out!

- **Email**: anzhaochong@outlook.com
- **Join in our Communication Group (WeChat)**:
<div style="text-align: left;">
    <img src="https://files.mdnice.com/user/67517/068822c4-cece-4ac5-b1db-5c138a91a718.png" width="200"/>
</div>