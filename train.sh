#!/bin/bash

accelerate launch --config_file accelerate_configs/multi_gpu.yaml \
    --num_processes=8 \
    --main_process_port 19961  \
    scripts/train_sga_grpo.py \
    --config config/sga_grpo.py:geneval_sga_grpo