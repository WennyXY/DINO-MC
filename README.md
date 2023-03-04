# DINO-MC: Self-supervised Contrastive Learning for Remote Sensing Imagery with Multi-sized Local Views

PyTorch implementation and pretrained models for DINO-MC and DINO-TP. For details, please see our paper. **DINO-MC: Self-supervised Contrastive Learning for Remote Sensing Imagery with Multi-sized Local Views**.

## Pretrained models
Our models are pre-trained on SeCo-100K, and we list their k-nn and linear probing accuracy on EuroSAT.
You can download the full checkpoint of pre-trained model with training infomation as well as weights of teacher and student networks used for the downstream tasks.

<table>
  <tr>
    <th>model</th>
    <th>arch</th>
    <th>params</th>
    <th>k-nn</th>
    <th>linear</th>
    <th>download</th>
  </tr>
  <tr>
    <td>DINO-MC</td>
    <td>ViT-S/16</td>
    <td>21M</td>
    <td>93.41%</td>
    <td>94.09%</td>
    <td><a href="https://drive.google.com/file/d/18RqKqZYzigOjwbyNzLsys8bmwqxrNhyt/view?usp=share_link">pre-trained ckpt</a></td>
  </tr>
  <tr>
    <td>DINO-MC</td>
    <td>ResNet-50</td>
    <td>23M</td>
    <td>93.94%</td>
    <td>95.59%</td>
    <td><a href="https://drive.google.com/file/d/1Tku4QoQDc3BU1BOr8PzQWFPyVStUDsVE/view?usp=share_link">pre-trained ckpt</a></td>
  </tr>
  <tr>
    <td>DINO-MC</td>
    <td>WRN-50</td>
    <td>69M</td>
    <td>95.65%</td>
    <td>95.70%</td>
    <td><a href="https://drive.google.com/file/d/1WlNDoks3Uo_Al5pUHWrhQpljDrt4Ip__/view?usp=share_link">pre-trained ckpt</a></td>
  </tr>
  <tr>
    <td>DINO-MC</td>
    <td>Swin-t</td>
    <td>28M</td>
    <td>93.22%</td>
    <td>90.54%</td>
    <td><a href="https://drive.google.com/file/d/1rod3PxdZ2OGqNJxLp5CAtXm7vIsLO7us/view?usp=share_link">pre-trained ckpt</a></td>
  </tr>

  <tr>
    <td>DINO-TP</td>
    <td>ViT-S/16</td>
    <td>21M</td>
    <td>93.15%</td>
    <td>93.89%</td>
    <td><a href="https://drive.google.com/file/d/1BIRR56wCwTDlB4_eQTA0DpYYPHACHfxN/view?usp=share_link">pre-trained ckpt</a></td>
  </tr>
  <tr>
    <td>DINO-TP</td>
    <td>ResNet-50</td>
    <td>23M</td>
    <td>79.05%</td>
    <td>86.70%</td>
    <td><a href="https://drive.google.com/file/d/1mHR9uv5G7-9FpEzGBvdEOcJcWbnHOGEV/view?usp=share_link">pre-trained ckpt</a></td>
  </tr>
  <tr>
    <td>DINO-TP</td>
    <td>WRN-50</td>
    <td>69M</td>
    <td>86.27%</td>
    <td>88.15%</td>
    <td><a href="https://drive.google.com/file/d/1MoclNnRlSGOKhudm5lreDYSxYJqciQar/view?usp=share_link">pre-trained ckpt</a></td>
  </tr>
  <tr>
    <td>DINO-TP</td>
    <td>Swin-t</td>
    <td>28M</td>
    <td>92.83%</td>
    <td>91.94%</td>
    <td><a href="https://drive.google.com/file/d/1E00rYPB2wFvnq7exmQwVRe1koQ98BECL/view?usp=share_link">pre-trained ckpt</a></td>
  </tr>
</table>


## Training
Our code is based on <a href="https://github.com/facebookresearch/dino">DINO</a> and <a href="https://github.com/ServiceNow/seasonal-contrast">SeCo</a>. 
If you want to pre-train DINO-MC based on your datasets: 
1. download DINO code as well as this code
2. replace main_dino.py with main_dino_mc.py and put data_process in the root directory.