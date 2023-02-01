# Spiderino Sensors Modeling

## Introduction

This work is supported by the Smart Grids team in Alpen Adria University working on Spiderino Robots. Spiderino Robots are an innovative way to experience robotics and particularly swarm robotics.
The goal of this project was to provide a model for IR sensors used on the Spiderino.

* Diagram of the different components of the platform :
<p align="center">
  <img src="https://user-images.githubusercontent.com/8127716/216053527-aebf1bbc-884a-41e8-b5a1-37f8534dfe5e.PNG" width="700" height="700*1.92" />
</p>

* Camera-vision processing steps :
<p align="center">
  <img src="https://user-images.githubusercontent.com/8127716/216056568-d6a8cf55-993a-4dc8-9dc1-e4f21623e5ab.PNG" width="900" height="900*1.92" />
</p>

## Implemented functionalities

* Camera-based data collection method.
* Dataset creation.
* Modeling approaches.

## Reference

TODO : Link to report

## Software prerequisites

* Python 3.9
* Scientific libraries (scikit-learn, numpy, pandas, matplotlib, seaborn, ...)
* Tensorflow + Keras

## Hardware prerequisites

* 1 x Modified Spiderino (for sensors data values). 
* 1 x ESP8266 to receive sensor values from the Spiderino and give them to PC via serial. 
* 1 x HDMI camera (used here is GOOKAM 4K WIFI camera)

## Installation

* Clone/Fetch repository

## Usage

* ``/3d-prints`` 3D prints models used in the process of designing the Modified Spiderino
* ``/camera-calibration`` camera calibration thingies (checker board pictures + calibration script)
* ``/master-script`` main scripts for dataset creation (see Dataset Creation section for details)
* ``/modeling/Regression-colab.ipynb`` script for dataset modeling
* ``/modeling/datasets/dataset_0312`` main/final dataset used for experiments


## Dataset Creation
General software workflow of scripts under ``/master-script`` is explained below :

<p align="center">
  <img src="https://user-images.githubusercontent.com/8127716/216052386-f9beb962-9445-41fd-a59b-1cbf4a598136.PNG" width="700" height="700*1.92" />
</p>

## Known Bugs

* bug 1
* bug 2

## Notes

* Note 1
* Note 2
