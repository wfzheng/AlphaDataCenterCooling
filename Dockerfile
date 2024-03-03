FROM ubuntu:18.04

# Install required packages
RUN 	apt-get update && \
    	apt-get install -y \
    	wget \
    	libgfortran4

# Set the user to root to have permissions to install packages
USER root
# Install the libc6 package to update the GLIBC version, necessary for compiling FMU requirements
RUN echo "deb http://th.archive.ubuntu.com/ubuntu jammy main" >> /etc/apt/sources.list
RUN apt-get update
RUN apt-get install -y libc6

# Install commands for Spawn
ENV SPAWN_VERSION=0.3.0-8d93151657
RUN wget https://spawn.s3.amazonaws.com/custom/Spawn-$SPAWN_VERSION-Linux.tar.gz \
    && tar -xzf Spawn-$SPAWN_VERSION-Linux.tar.gz \
    && ln -s /Spawn-$SPAWN_VERSION-Linux/bin/spawn-$SPAWN_VERSION /usr/local/bin/

# Create new user
RUN 	useradd -ms /bin/bash user
USER user
ENV 	HOME /home/user

# Download and install miniconda and pyfmi
RUN 	cd $HOME && \
	wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -O $HOME/miniconda.sh && \
	/bin/bash $HOME/miniconda.sh -b -p $HOME/miniconda && \
	. miniconda/bin/activate && \
	conda update -n base -c defaults conda && \
	conda create --name pyfmi3 python=3.10 -y && \
	conda activate pyfmi3 && \
	conda install -c conda-forge pyfmi=2.11 -y && \
	pip install flask-restful==0.3.9 werkzeug==2.2.3 && \
    pip install torch &&\
	conda install pandas==1.5.3 flask_cors==3.0.10 matplotlib==3.7.1 requests==2.28.1

WORKDIR $HOME


ENV PYTHONPATH $PYTHONPATH:$HOME

CMD . miniconda/bin/activate && conda activate pyfmi3 && python restapi.py && bash

EXPOSE 5000