import subprocess

with open('scripts/experiments.txt') as f:
    cmd_list = f.read().splitlines() 

for cmd in cmd_list[1:]:
    subprocess.call(cmd, shell=True)