# OpenGANForensics
Official PyTorch implementation of
["**Open Set Classification of GAN-based Image Manipulations via a ViT-based Hybrid Architecture**"](https://openaccess.thecvf.com/content/CVPR2023W/WMF/papers/Wang_Open_Set_Classification_of_GAN-Based_Image_Manipulations_via_a_ViT-Based_CVPRW_2023_paper.pdf). 

## Highlights & Research Contributions
- **Open-set handling**: Designed for classification scenarios where out-of-set classes may appear at inference, focusing on GAN-based image manipulations.
- **Hybrid ResNet + ViT architecture**: Combines a CNN backbone (ResNet50) with a ViT head; an optional localization branch captures both local and long-range dependencies.
- **Joint classification & localization**: Flags `--loc` and `--masks` enable mask supervision to highlight manipulated regions alongside image-level classification.
- **Flexible configuration**: Flags `--nodown`, `--patch_size`, `--dim`, `--depth`, `--head` let you tune feature resolution and model capacity to match GPU budgets.
- **Easy reproducibility**: Provided open/closed-set splits (configs.txt) and pretrained models enable quick benchmarking and comparison.

## 1. Requirements
### Environments
Before running the code, please configure your env following the requirement file.

### Datasets
We collected our own dataset using the code and models released by [PTI](https://github.com/danielroich/PTI).

### Quick setup checklist
1. Install dependencies: `pip install -r requirements.txt`.
2. Prepare data:
   - Facial attribute edit: place images/masks under `./data/` (or custom path) and update `labels_train_list.txt`, `labels_valid_list.txt`.
   - GAN attribution: use the `./dati/` structure (folder name as provided in the repo) with splits defined in `configs.txt`.
3. (Optional) Download pretrained models (link below) and place them in `./save_models/`.
4. Check GPU budget: batch size 32 suits ≥24GB VRAM; reduce batch if memory is limited.

## 2. Training and testing

**We provide two networks for training. The users can replace the dataset load function based on their own tasks.**

To train the models in our paper for classification and localization, run this command:

Run hybrid classification and localization network for facial attribute edit images
```
python main.py --save_models ./path_to_save_model --model resnet50 --classes 11 --loc --nodown --masks
```
Run classification network for GAN attribution task
```
python main.py --save_models ./path_to_save_model --model resnet50 --classes 3 --nodown
```
Note: if loc, masks and nodown are activated, the network is exactly the one we used in our work for facial attributes edit classification. Without --loc, it will be resnet50 + Vit network for GAN attribution task. Masks indicates loading ground truth mask for localization task.

For open set test, simply run the command:

Run hybrid classification and localization network for facial attribute edit images
```Open set test for facial attribute edit classification
python3 test_osr.py -m ./saved_model/resnet50ND_Vit2_S2_4_b32/**.pth --loc --nodown --pretrain --classes 11 --data_path ./data/
```
Run classification network for GAN attribution task
```Open set test for GAN attribution
python3 test_osr.py -m ./saved_model/resnet50ND_Vit2_S2_4_b32/**.pth  --nodown --pretrain --classes 3 --data_path ./dati/
```

### Usage tips for best results
- **Pick the right network configuration**: 
  - Use `--loc --masks --nodown` for facial attribute edit detection with localization.
  - Use `--nodown` (without `--loc`) for GAN attribution open-set classification.
- **Pretrain & fine-tune**: Add `--pretrain --weights_path <file>` for better initialization, especially with limited data.
- **Tune the ViT head**: Smaller `--patch_size` captures finer details; increase `--dim`, `--depth`, `--head` if GPU memory allows.
- **Balance the losses**: Adjust `--lambda_locs` and `--lambda_clss`; raise `lambda_locs` when high-quality masks are available.
- **Control overfitting**: Use `--patient` for early stopping; consider lowering `--lr` or increasing `--drop` if validation accuracy oscillates.
- **Evaluate open-set robustness**: Run `test_osr.py` with the dedicated open-set split (per `configs.txt`) to assess generalization.
### Pre-trained Model

You can find one of the pretrained models [here](https://drive.google.com/drive/folders/1tO_0PQvlSm_bbpe1zhyOnIF6kPZg2MX9?usp=drive_link)

resnet50ND_Vit2_S1_4_b32 for GAN attribution and resnet50_Vit2_S1_4_b32 for Facial attributes editing classification. “S1”, “S2” etc. indicates the in-set / out-set configurations, detailed [here](https://github.com/wangjun9276/OpenGANForensics/blob/main/dati/configs.txt)

In data/dati folders, you can find configs.txt which records the splits of the closed- and open-set.
## Citation
- If you find our work or the code useful, please consider citing our paper using:
```bibtex
@inproceedings{wang2023open,
  title={Open Set Classification of GAN-based Image Manipulations via a ViT-based Hybrid Architecture},
  author={Wang, Jun and Alamayreh, Omran and Tondi, Benedetta and Barni, Mauro},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={953--962},
  year={2023}
}
```

## Contact
- If you find a problem with the code please contact me
