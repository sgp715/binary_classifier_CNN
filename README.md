# Binary Classifier Convolutional Neural Network

# Install
* setup a virtualenv
```
$ virtualenv bcCNN
```
* activate the virtualenv
```
$ source bcCNN/bin/activate
```
* pip install -r requirements.txt

# Usage
* train the network with images from two directories
```
$ python bcCNN.py -train <path/to/images1> <path/to/images2> <dir/to/save/model> <model/category>
```
* test the accuracy on images from two directories
```
$ python bcCNN.py -test <path/to/images1> <path/to/images2>
```
* classify a single image
```
$ python bcCNN.py -classify <path/image/to/classify>
```
