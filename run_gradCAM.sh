#!/bin/bash

echo "start running"

nvidia-smi

python gradcam_APs.py -save_loc /usr/xtmp/IAIABL/gradCAM_imgs/view.png

echo "finish running"