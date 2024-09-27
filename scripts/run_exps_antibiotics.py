import subprocess

with open('scripts/experiments_antibiotics.txt') as f:
    cmd_list = f.read().splitlines() 

for cmd in cmd_list:
    subprocess.call(cmd, shell=True)