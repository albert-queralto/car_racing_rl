# Stage 1: Build stage
ARG PYTORCH_VERSION=23.10-py3

FROM nvcr.io/nvidia/pytorch:${PYTORCH_VERSION}

# Set environmental variables
ENV RUNNING_IN_DOCKER true
ENV DEBIAN_FRONTEND noninteractive

# OpenCV Version
ARG OPENCV_VERSION="4.9.0"

# Install build dependencies
RUN apt-get clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
        build-essential binutils \
        ca-certificates cmake cmake-qt-gui curl \
        dbus-x11 \
        ffmpeg \
        gdb gcc g++ gfortran git \
        tar \
        lsb-release \
        procps \
        manpages-dev \
        unzip \
        zip \
        wget \
        xauth \
        swig \
        software-properties-common \
        python3-pip python3-dev python3-numpy python3-distutils \
		python3-setuptools python3-pyqt5 python3-opencv \
        libboost-python-dev libboost-thread-dev libatlas-base-dev libavcodec-dev \
        libavformat-dev libavutil-dev libcanberra-gtk3-module libeigen3-dev \
        libglew-dev libgl1-mesa-dev libgl1-mesa-glx libglib2.0-0 libgtk2.0-dev \
        libgtk-3-dev libjpeg-dev libjpeg8-dev libjpeg-turbo8-dev liblapack-dev \
        liblapacke-dev libopenblas-dev libopencv-dev libpng-dev libpostproc-dev \
        libpq-dev libsm6 libswscale-dev libtbb-dev libtbb2 libtesseract-dev \
        libtiff-dev libtiff5-dev libv4l-dev libx11-dev libxext6 libxine2-dev \
        libxrender-dev libxvidcore-dev libx264-dev libgtkglext1 libgtkglext1-dev \
        libvtk9-dev libdc1394-dev libgstreamer-plugins-base1.0-dev \
        libgstreamer1.0-dev libopenexr-dev libqt5x11extras5\
        openexr \
        pkg-config \
        qv4l2 \
        v4l-utils \
        zlib1g-dev \
        locales \
        && locale-gen en_US.UTF-8 \
        && LC_ALL=en_US.UTF-8 \
        && rm -rf /var/lib/apt/lists/* \
        && apt-get clean

# Install OpenCV by compiling it
WORKDIR /opencv
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
    && wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip \
    && unzip opencv.zip \
    && unzip opencv_contrib.zip \
    && mv opencv-${OPENCV_VERSION} opencv \
    && mv opencv_contrib-${OPENCV_VERSION} opencv_contrib
RUN mkdir /opencv/opencv/build
WORKDIR /opencv/opencv/build

# Change the last line to be the number of cores you allocate to docker - j4 for 4 cores
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D WITH_CUDA=ON \
    -D CUDA_VERSION=12.0 \
    -D WITH_CUBLAS=ON \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D BUILD_TIFF=ON \
    -D BUILD_opencv_java=OFF \
    -D WITH_VTK=OFF \
    -D BUILD_TESTS=OFF \
    -D WITH_IPP=ON \
    -D WITH_TBB=ON \
    -D WITH_EIGEN=ON \
    -D WITH_V4L=ON \
    -D BUILD_PERF_TESTS=OFF \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D BUILD_opencv_python2=OFF \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
    -D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
	-D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    -D INSTALL_PYTHON_EXAMPLES=ON \
	-D INSTALL_C_EXAMPLES=ON \
	-D OPENCV_ENABLE_NONFREE=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D PYTHON3_EXECUTABLE=$(which python3) \
    -D PYTHON_DEFAULT_EXECUTABLE=$(which python3) \
	-D OPENCV_EXTRA_MODULES_PATH=/opencv/opencv_contrib/modules \
	-D BUILD_EXAMPLES=ON .. \
    && make -j$(nproc) && make install && ldconfig

# Set the working directory to $APP_USER_HOME
WORKDIR $APP_USER_HOME

# Copy requirements.txt to the working directory
COPY requirements.txt .

# Install the Python dependencies on the virtual environment
RUN python3 -m pip install --upgrade pip \
	&& pip install --no-cache-dir -r requirements.txt \
	&& rm requirements.txt

# Set the LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH /opt/hpcx/ucx/lib:$LD_LIBRARY_PATH

WORKDIR $APP_USER_HOME

RUN chsh -s ~/.bashrc
ENV SHELL /bin/bash