#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=Crosslinear
pred_lens=(96 192 336 720)
correlation_weights=(0.0 0.2 0.4 0.6 0.8 1.0)
pos_weights=(0.0 0.2 0.4 0.6 0.8 1.0)

for pred_len in "${pred_lens[@]}"; do
  for w1 in "${correlation_weights[@]}"; do
    for w2 in "${pos_weights[@]}"; do
      model_id="ETTh1_96_${pred_len}_${w1}_${w2}"
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id "$model_id" \
        --model "$model_name" \
        --data ETTh1 \
        --features MS \
        --seq_len 96 \
        --label_len 48 \
        --pred_len "$pred_len" \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --itr 1 \
        --correlation_weight "$w1" \
        --pos_weight "$w2"
    done
  done
done