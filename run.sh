#!/bin/bash

# run.sh

python utils/make-data.py > graph

make -s clean; make -s
./bin/auction > res