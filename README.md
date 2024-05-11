# Deep Reinforcement Learning for Car Racing

## Overview
This repository contains the implementation of several reinforcement learning algorithms to solve the [Car Racing](https://gymnasium.farama.org/environments/box2d/car_racing/)

The algorithms implemented are:
- [Deep Q-Learning](https://arxiv.org/abs/1312.5602)
- [Dueling Deep Q-Learning](https://arxiv.org/abs/1511.06581)
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- Variational Quantum Circuit Deep Q-Learning

The implementation is done using the [PyTorch](https://pytorch.org/) and [Pennylane](https://pennylane.ai/) libraries.

## Installation
The project is developed using Python 3.10.12 and includes a containarized environment with Docker that can be deployed using the following files:
- *Dockerfile*: contains the instructions to build the Docker image to create a GPU enabled container.
- *docker-compose.yml*: contains the instructions to create the container, e.g., services, volumes, GPUs, etc.
- *requirements.txt*: contains the list of Python packages required to run the project.

Prior to running the container, Docker and Docker Compose must be installed, as well as NVIDIA Container Toolkit in order to use GPUs. Finally, the developed tool uses [OpenCV](https://opencv.org/) to render the environment, which also requires a set up in the container.

The following links provide instructions on how to set a development environment with GPU access using Docker in Visual Studio Code, as well as how to install OpenCV in the container:
- [Setting Up Docker on Ubuntu](https://medium.com/@albertqueralto/setting-up-docker-on-ubuntu-511fa5b5e897)
- [Enabling CUDA Capabilities in Docker Containers](https://medium.com/@albertqueralto/enabling-cuda-capabilities-in-docker-containers-51a3566ad014)
- [Creating a VS Code development environment for Deep Learning](https://medium.com/@albertqueralto/creating-a-vs-code-development-environment-for-deep-learning-91b74621685e)
- [Installing OpenCV within Docker containers for Computer Vision and Development](https://medium.com/@albertqueralto/installing-opencv-within-docker-containers-for-computer-vision-and-development-a93b46996520)

The following commands can be used to build the Docker image and run the container from the root directory of the project:
```bash
docker-compose -f docker-compose.yml up --build -d
```