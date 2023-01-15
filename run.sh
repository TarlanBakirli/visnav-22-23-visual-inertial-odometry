#!/bin/sh

cd build
make odometry
cd ../
build/odometry --dataset-path data/MH_01_easy/mav0/