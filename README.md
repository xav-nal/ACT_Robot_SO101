# ACT_Robot_SO101
Implementation in Pytorch of Action Chunking with Transformers on the SO101 robot



### Linux config
make sure to add to you linux group: dialout and input for usb and keyboard events
sudo usermod -aG dialout,input $USER
newgrp input
newgrp dialout

rerun analytics disable



### Commands to train

python src/act_train.py

tensorboard --logdir logs




