# OpenGANForensics
Official PyTorch implementation of
["**Open Set Classification of GAN-based Image Manipulations via a ViT-based Hybrid Architecture**"](https://openaccess.thecvf.com/content/CVPR2023W/WMF/papers/Wang_Open_Set_Classification_of_GAN-Based_Image_Manipulations_via_a_ViT-Based_CVPRW_2023_paper.pdf). 

## Highlights & Research Contributions
- **Xử lý open-set**: Thiết kế cho kịch bản phân loại với các lớp ngoài-tập xuất hiện ở suy diễn, tập trung vào thao tác ảnh GAN.
- **Kiến trúc lai ResNet + ViT**: Kết hợp backbone CNN (ResNet50) với head ViT; tùy chọn nhánh localization giúp khai thác đặc trưng cục bộ và quan hệ dài hạn.
- **Phân loại & định vị đồng thời**: Tham số `--loc` và `--masks` kích hoạt nhánh mask supervision để chỉ vùng thao tác, ngoài phân loại toàn ảnh.
- **Cấu hình linh hoạt**: Tham số `--nodown`, `--patch_size`, `--dim`, `--depth`, `--head` cho phép điều chỉnh độ phân giải đặc trưng và năng lực mô hình phù hợp tài nguyên GPU.
- **Tái lập dễ dàng**: Cung cấp split open/closed-set (configs.txt) và model pretrained, giúp thử nghiệm nhanh và so sánh.

## 1. Requirements
### Environments
Before running the code, please configure your env following the requirement file.

### Datasets
We collected our own dataset using the code and models released by [PTI](https://github.com/danielroich/PTI).

### Quick setup checklist
1. Cài đặt phụ thuộc: `pip install -r requirements.txt`.
2. Chuẩn bị dữ liệu:
   - Facial attribute edit: đặt ảnh/mask theo đường dẫn `./data/` (hoặc custom) và cập nhật `labels_train_list.txt`, `labels_valid_list.txt`.
   - GAN attribution: dùng cấu trúc `./dati/` với file split trong `configs.txt`.
3. (Tuỳ chọn) Tải model pretrained (link bên dưới) và đặt tại `./save_models/`.
4. Kiểm tra GPU: batch 32 mặc định phù hợp GPU ≥24GB; giảm batch nếu thiếu VRAM.

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
- **Chọn đúng cấu hình mạng**: 
  - `--loc --masks --nodown` cho nhận diện + định vị chỉnh sửa thuộc tính khuôn mặt.
  - `--nodown` (không `--loc`) cho GAN attribution open-set.
- **Pretrain & fine-tune**: Thêm `--pretrain --weights_path <file>` để khởi tạo tốt hơn, đặc biệt khi dữ liệu nhỏ.
- **Điều chỉnh ViT**: `--patch_size` nhỏ cho chi tiết mịn; tăng `--dim`, `--depth`, `--head` nếu GPU đủ lớn.
- **Cân bằng loss**: Điều chỉnh `--lambda_locs` và `--lambda_clss`; tăng `lambda_locs` khi mask ground truth chất lượng cao.
- **Giảm overfitting**: Dùng `--patient` cho early stopping; cân nhắc giảm `--lr` hoặc tăng `--drop` nếu valid acc dao động.
- **Đánh giá open-set**: Sử dụng `test_osr.py` với tập open-set tách biệt (theo `configs.txt`) để kiểm tra khả năng tổng quát hóa.
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
