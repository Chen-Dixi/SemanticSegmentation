# SemanticSegmentation
在Cityscapes数据集上的Pytorch语义分割实践

## Semantic Segmentation Toolbox
an Awesome PyTorch's Semantic Segmentation Toolbox: [BLSeg]

## Requirement

* Python 3
* PyTorch >= 1.0.0
* [dixitool]
* tqdm
* [cityscapesscripts]
* Pillow

## Visualization

只运行了4个epoch看结果，主要目的是上手语义分割

| | Original Image  |  Predict Mask  |
| :--: | :------------: | :------------: |
|Epoch:1|   ![image]   | ![predict1] |
|Epoch:4|   ![image]   | ![predict2] |



---

[predict1]:assets/result/predict1epoch1.png
[predict2]:assets/result/predict1epoch3.png
[BLSeg]:https://github.com/linbo0518/BLSeg
[dixitool]:https://github.com/Chen-Dixi/dixitool
[cityscapesscripts]:https://github.com/mcordts/cityscapesScripts
[image]:assets/result/frankfurt_000000_000576_leftImg8bit.png