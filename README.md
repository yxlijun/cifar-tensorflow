## Train and test CIFAR10 with tensorflow
#### introduction 
> various nets implement train and test in cifar10 dataset with tensorflow,the nets include `VGG`,`Resnet`,`Resnext`,`mobilenet`,`SENet`,`xception` and so on

 
**** 

#### Prepare data,Testing,Training
1. Clone the cifar-tensorflow repository  
```
 git https://github.com/yxlijun/cifar-tensorflow
```
2. Prepare data,run folloing code,generate cifar10_data directory  
```
python tools/cifar10_download_and_extract.py
```
3. training  
```
 python train.py --net vgg16
```
4. testing  
```
python test.py --net vgg16
```


****
####Accuracy

| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG11](https://arxiv.org/abs/1409.1556)              | 92.64%      |
| [VGG13](https://arxiv.org/abs/1409.1556)          | 93.02%      |
| [VGG16](https://arxiv.org/abs/1409.1556)          | 93.62%      |
| [VGG19](https://arxiv.org/abs/1409.1556)         | 93.75%      |
| [Resnet20](https://arxiv.org/abs/1512.03385)       | 94.43%      |
| [Resnet32](https://arxiv.org/abs/1512.03385)  | 94.73%      |
| [Resnet44](https://arxiv.org/abs/1512.03385)  | 94.82%      |
| [Resnet56](https://arxiv.org/abs/1512.03385)       | 95.04%      |
| [Xception](https://arxiv.org/abs/1610.02357)    | 95.11%      |
| [MobileNet](https://arxiv.org/abs/1704.04861)             | 95.16%      |
| [DensetNet40_12](https://arxiv.org/abs/1608.06993) | 94.24% |
| [DenseNet100_12](https://arxiv.org/abs/1608.06993)| 95.21%  |
| [DenseNet100_24](https://arxiv.org/abs/1608.06993)| 95.21%  |
| [DenseNet100_24](https://arxiv.org/abs/1608.06993)| 95.21%  |
| [ResNext50](https://arxiv.org/abs/1611.05431)| 95.21%  |
| [ResNext101](https://arxiv.org/abs/1611.05431)| 95.21%  |
| [SqueezeNetA](https://arxiv.org/abs/1602.07360)| 95.21%  |
| [SqueezeNetB](https://arxiv.org/abs/1602.07360)| 95.21%  |
| [SE_Resnet_50](https://arxiv.org/abs/1709.01507)| 95.21%  |
| [SE_Resnet_101](https://arxiv.org/abs/1709.01507)| 95.21%  |


### Net implement
- [x] VGG
- [x] ResNet
- [x] DenseNet
- [x] mobileNet
- [x] ResNext
- [x] Xception
- [x] SeNet
- [x] SqueenzeNet 