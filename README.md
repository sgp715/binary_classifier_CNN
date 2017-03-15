# Binary Classifier Convolutional Neural Network

# Install
* First, get Docker and if you don't have Docker, get with the times

* Next, let's build us a Docker image
```
$ docker build -t bccnn_image .
```

# Usage
* To start that image...we have to run a kind of sucky command
```
$ docker run -it -v $(pwd)/classifiers:/bccnn/classifiers -v $(pwd)/images:/bccnn/images bccnn_image /bin/bash
```
* You are now inside the docker container and can start training and testing to your hearts content and the models you train will persist outside of Docker
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
