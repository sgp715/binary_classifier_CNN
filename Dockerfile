FROM gcr.io/tensorflow/tensorflow

RUN apt-get update && apt-get install -y
RUN apt-get install pkg-config -y
RUN apt-get install python-pip -y
RUN apt-get install python-tk -y

WORKDIR /bccnn
COPY . /bccnn

RUN pip install -r requirements.txt
