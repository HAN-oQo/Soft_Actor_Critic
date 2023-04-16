FROM nvidia/cuda:11.6.1-cudnn8-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y software-properties-common
RUN apt-get install -y lsb-release 

RUN gpg --list-keys
RUN gpg --no-default-keyring --keyring /usr/share/keyrings/deadsnakes.gpg --keyserver keyserver.ubuntu.com --recv-keys F23C5A6CF475977595C89F51BA6932366A755776
RUN echo "deb [signed-by=/usr/share/keyrings/deadsnakes.gpg] https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/python.list

RUN apt -y update
RUN apt install -y python3.8
RUN ln -sf /usr/bin/python3.8 /usr/bin/python
RUN ln -sf /usr/bin/python3.8 /usr/bin/python3
RUN apt-get install -y python3-pip
RUN apt-get install -y ffmpeg
RUN apt-get install -y libsndfile1-dev

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y vim tmux

WORKDIR /app

RUN python -m pip install --upgrade pip
RUN python -m pip install --upgrade setuptools wheel

COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r requirements.txt
# RUN python -m pip install pybullet-svl

COPY entrypoint.sh /var/tmp
CMD bash -E /var/tmp/entrypoint.sh

#  docker build . -t gym_hq