# ResNet3D-TF2
This is a python code of ResNet3D model based on tensorflow 2.3.0 via subclassing. 
## environment
```
python 3
tensorflow 2.3.0
```
## paper
Chen, Yunpeng, Yannis Kalantidis, Jianshu Li, Shuicheng Yan, and Jiashi Feng. "Multi-fiber networks for video recognition." In Proceedings of the european conference on computer vision (ECCV), pp. 352-367. 2018.
## input data
The input data format is default as NxHxWxDxC, namely ```data_format="channels_last"```. 
## output data
The output is a batch of 2-node logits, namely pre-softmax. 
