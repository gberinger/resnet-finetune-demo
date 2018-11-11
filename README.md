# resnet-finetune-demo

<a href="url">
<img align="right" height="250" src="http://ml.cta.ai/blog/resnet-finetune-demo/studiomusic.png">
</a>

A simple experiment with finetuning Resnet-152 in Keras for classifying indoor places images from [MIT Indoor-67](http://web.mit.edu/torralba/www/indoor.html) dataset. This method achieves 73% accuracy on the test set.

Detailed description of this experiment can be found here: http://www.cta.ai/en/publications/02

The code is built on Keras 2.0 (ver. 2.0.4) on TensorFlow backend (ver. 1.2.0-rc2) using Python 3.6, but should work
on newer Keras and Tensorflow versions.

#### Environment

To prepare the environment, install the requirements:

`pip install -r requirements.txt`

Personally, I suggest installing [Anaconda](https://www.anaconda.com/) for this and future tasks, as it contains a lot 
of data science packages and allows to create many standalone environments. If you use Anaconda, you just need to 
install `tensorflow` and `Keras`  with pip or conda (CPU versions are good enough for this task).

#### Data

To run this experiment you should download the data package which is available 
[here](http://ml.cta.ai/blog/resnet-finetune-demo/resnet-finetune-demo-data-package.zip). 
Unpack it in this repository for compliance with the code. The data package consists of several elements:
- [Indoor-67](http://web.mit.edu/torralba/www/indoor.html) images, split into train, val and test subsets
- ImageNet-trained Resnet-152 weights, acquired from [here](https://github.com/flyyufelix/cnn_finetune)
- Text file mapping ImageNet ids to class names, acquired from [here](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)
- Cached features from Resnet-152 model and labels for images in train, val and test subsets

#### Method summary

This method simply freezes the entire Resnet-152 model and retrains the last fully-connected classification layer from scratch. It achieves 73% accuracy, which is only 6% less than state-of-the-art as for 2016 (see the chart below).

<p align="center"><a href="url">
<img height="250" src="http://ml.cta.ai/blog/resnet-finetune-demo/chart.png">
</a></p>


##### References

[1] Quattoni, Ariadna, and Antonio Torralba. "Recognizing indoor scenes." Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on. IEEE, 2009.

[2] Zhou, Bolei, et al. "Learning deep features for scene recognition using places database." Advances in neural information processing systems. 2014.

[3] Khan, Salman H., et al. "A discriminative representation of convolutional features for indoor scene recognition." IEEE Transactions on Image Processing 25.7 (2016): 3372-3383.

[4] Herranz, Luis, Shuqiang Jiang, and Xiangyang Li. "Scene recognition with CNNs: objects, scales and dataset bias." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.


#### Modules outline:
- resnet-demo.py - demo app testing ImageNet trained Resnet-152 on images from clipboard
- build_features.py - creates a cache of Resnet features over the dataset
- train.py - builds fully-connected classification layer and trains it over the cached features
- test.py - test the trained classifier
- test-demo.py - demo app classifying test images one by one and showing outputs
- helper.py - some helping functions
- resnet/resnet152.py - Resnet-152 model acquired from [here](https://github.com/flyyufelix/cnn_finetune)
