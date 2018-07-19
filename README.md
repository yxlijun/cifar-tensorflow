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
| [Resnet](https://arxiv.org/abs/1801.04381)       | 94.43%      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%      |
 
### Net implement
- [x] VGG
- [x] ResNet
- [x] DenseNet
- [x] mobileNet
- [x] ResNext
- [x] Xception
- [x] SeNet
- [x] SqueenzeNet 