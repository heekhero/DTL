#!/usr/bin/env bash

CONFIG=$1

random_port=$((RANDOM % 65535 + 1024))

# 检查端口是否被占用
while [[ "$(netstat -tuln | grep -w "$random_port")" != "" ]]; do
    random_port=$((RANDOM % 65535 + 1024))
done

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=$random_port \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
