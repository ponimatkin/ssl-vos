#!/bin/bash
wget https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip
unzip models.zip

mv *.pth models/

cd models/
wget https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth