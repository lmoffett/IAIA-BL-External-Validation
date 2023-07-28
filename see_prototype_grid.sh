#!/bin/bash

nvidia-smi

srun -u python3 vis_protos.py
