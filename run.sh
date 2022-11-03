#!/bin/bash

conda env create -f environment.yml && \
conda activate hateop-train && \
python train.py && \
python eval.py