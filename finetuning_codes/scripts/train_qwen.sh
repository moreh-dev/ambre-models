#!/bin/bash

START_TIME=$(TZ="Asia/Seoul" date)
current_time=$(date +"%y%m%d_%H%M%S")

TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file /root/poc/finetuning_codes/config.yaml \
    /root/poc/finetuning_codes/train.py \
    --model /root/poc/pretrained_models/Qwen-14B \
    --dataset alespalla/chatbot_instruction_prompts \
    --lr 0.0001 \
    --train-batch-size 128 \
    --eval-batch-size 16 \
    --num-epochs 5 \
    --max-steps 20 \
    --log-interval 20 \
    --save-path /root/poc/checkpoints/qwen_finetuned_$current_time \
    |& tee /root/poc/finetuning_codes/logs/qwen_finetune_$current_time.log

echo "Start: $START_TIME"
echo "End: $(TZ="Asia/Seoul" date)"
