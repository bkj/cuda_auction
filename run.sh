#!/bin/bash

# run.sh

# python utils/make-data.py --dim 1024 > graph

make -s clean; make -s
nvprof ./bin/auction > res