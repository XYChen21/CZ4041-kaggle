# CZ4041-kaggle
Each model is saved based on best val acc and val loss. Link to models by clicking the AUC number  
csv files in `submission` folder
|ID| name | best_acc_AUC| best_loss_AUC|
|--| ---- | ----------- | ------------ |
|m1| m1_vit_fixed | [0.772](https://drive.google.com/file/d/1EAF-Z3yOO8TI9I73JZ6s6tk9C6JPsdK0/view?usp=share_link) | [0.772](https://drive.google.com/file/d/1TiuLANgCMneaSBcoVnpDa3EYqfZ5SgUF/view?usp=share_link) |[Drive](https://breakdance.github.io/breakdance/)
|m2| m2_vit_finetune | [0.841](https://drive.google.com/file/d/1pP7gJVvwtid48ep5be7ceJ96ZV_S_kh8/view?usp=share_link) | [0.841](https://drive.google.com/file/d/1TR2JNEP__WrSI6aoav3SlUt8AtYaJcHU/view?usp=share_link)|
|m3| m3_vit_randomtune | [**0.884**](https://drive.google.com/file/d/1-1cyWw3yRScBAZhXQtPAuAWQDtncyzRb/view?usp=sharing) | [**0.884**](https://drive.google.com/file/d/1BDV-NRcHoAWmsAcd_j-6-a2f9FmGNMd5/view?usp=sharing) |
|m4| m4_vit_randomtune2 | [**0.895**](https://drive.google.com/file/d/1yWE8AJVqGqLroIdj_yD3b0Q81zBvj8bR/view?usp=share_link) | [**0.878**](https://drive.google.com/file/d/1zLAQzXTD9XfnhNfLyPdiurY_9tBMPsKm/view?usp=sharing) |
|m5| m5_vitcnn_attnpool | [**0.874**](https://drive.google.com/file/d/1dBSRHT1_N7KM--KktLACAXyRUFOv09tO/view?usp=sharing)  | [**0.871**](https://drive.google.com/file/d/17DE2CDeJHSG5M3tBQVoLqh439v_-i3oS/view?usp=sharing) |
|m6| m6_vit_contrastive | [0.813](https://drive.google.com/file/d/1-AJC7L0XUshY1zlx6Fvh_X9qLkbATcj_/view?usp=sharing) | [0.813](https://drive.google.com/file/d/1vBs_8QgHO6itU14Xw_2iRfQVgd89NjsG/view?usp=sharing) |
|m7| m7_vit_crossattn | [0.67](https://drive.google.com/file/d/15Kn7i9Fjm6L6tDMQ_Cocrl8HKIlBBVts/view?usp=share_link) | [0.67](https://drive.google.com/file/d/1Nwp_vLsfldud28YOuUvbVD_L0goRsidP/view?usp=share_link) |

|ID| Explanation |
|--| ----------- |
|m1| Fixed ViT weights|
|m2| Finetune last attention layer of ViT|
|m3| Randomly choose a layer to finetune for 10 epochs, creating new optimizer each time|
|m4| Randomly choose a layer to finetune for 10 epochs, by only chaning `.require_grad_(True/False)`|
|m5|ViT with techniques in m4 + ResNet with AttentionPool|
|m6|ViT with contrastive loss (all layers trained)|
|m7|ViT with added Linear projections in each Attention|

Did a simple ensemble by trying combinations of csv results, should change to other ways later. AUC 0.9 on kaggle