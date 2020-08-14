#!/bin/bash

export DGLBACKEND=tensorflow

python3 train.py \
	--dataset reddit \
	--gpu 0 \
	--self-loop \
	--n-hidden 512 \
	--n-layers 4 \
	--n-epochs 5

