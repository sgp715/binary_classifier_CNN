# Binary Classifier Convolutional Neural Network

# Install
* First, get Docker and if you don't have Docker, get with the times

* Next, let's build us a Docker image
```
$ docker build -t bccnn_image .
```

# Usage
* Make 2 directories one called images and another called classifiers
    * The first one will hold the set of images that you want to train on
    * The way I would structure this folder if I was going to train a formal classifier would be to put the formal images in images/formal/positive and images/formal/negative
* To start that image we can run this command which will also allow all our changes made to persist in the docker image
```
$ docker run -it -v $(pwd):/bccnn bccnn_image /bin/bash
```
* You are now inside the docker container and can start training and testing to your hearts content and the models you train will persist outside of Docker as long as you save them inside the classifiers directory
* Train the network with images from two directories
```
$ python bcCNN.py -train <path/to/images1> <path/to/images2> <dir/to/save/model> <model/category>
```
* Here path <path/to/images1> corresponds to images/formal/positive and <path/to/images2> corresponds to images/formal/negative from the example above
* <dir/to/save/model> corresponds to where we want the model we train to be saved which in our case is classifiers
* And <model/category> is what we want the name of the model to be (for us formal)
* Test the accuracy on images from two directories
```
$ python bcCNN.py -test <path/to/images1> <path/to/images2>
```
* Classify a single image
```
$ python bcCNN.py -classify <path/image/to/classify>
```
