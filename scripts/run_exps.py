import subprocess
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"


with open('scripts/experiments_antibiotics.txt') as f:
    cmd_list = f.read().splitlines() 

for cmd in cmd_list[:2]:
    subprocess.call(cmd, shell=True)