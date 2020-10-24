# ResNet3D-TF2
This is a python code of ResNet3D model based on tensorflow 2.3.0 via subclassing. 
## environment
```
python 3
tensorflow 2.3.0
```
## reference
Hara, Kensho, Hirokatsu Kataoka, and Yutaka Satoh. "Can spatiotemporal 3d cnns retrace the history of 2d cnns and imagenet?." In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pp. 6546-6555. 2018.
## input data
The input data format is default as NxHxWxDxC, namely ```data_format="channels_last"```. 
## output data
The output is a batch of 2-node logits, namely pre-softmax. 
