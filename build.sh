#!/bin/bash

# mkdir -p ./host_output
# mkdir -p ./host_upload

# test
docker build -t dkr.tw/sgis/gcn image
docker run -it --rm --gpus=all --shm-size="64g" -p 7676:7676 dkr.tw/sgis/gcn

# local version with two volumes
# docker build -t dkr.tw/sgis/gcn image
# docker run -it --rm --gpus=all --shm-size="64g" -p 7676:7676 -v "$(pwd)/host_upload:/workspace/Mask2Former/upload" -v "$(pwd)/host_output:/workspace/Mask2Former/output" dkr.tw/sgis/gcn