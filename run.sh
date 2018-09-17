#!/bin/bash

# run.sh

# compile
nvcc -w non_atomic_main.cu -o main -gencode=arch=compute_60,code=\"sm_60,compute_60\";

# run
./main > res