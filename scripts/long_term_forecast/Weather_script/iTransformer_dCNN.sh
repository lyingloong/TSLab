#!/bin/bash
exec > output_iTransformer_dCNN.txt 2>&1
export CUDA_VISIBLE_DEVICES=1

model_name=iTransformer_dCNN

seq_len=96
enc_in=21
pred_lens=96
t_kernel_sizes=(1 15 31 63 95)
v_kernel_sizes=(1 7 15 21)

for t_kernel_size in "${t_kernel_sizes[@]}"; do
  for v_kernel_size in "${v_kernel_sizes[@]}"; do
    model_id="weather_96_${pred_lens}_${t_kernel_size}_${v_kernel_size}"
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/weather/ \
      --data_path weather.csv \
      --model_id "$model_id" \
      --model $model_name \
      --data custom \
      --features MS \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 96 \
      --e_layers 3 \
      --d_layers 1 \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --time_kernel_size "$t_kernel_size" \
      --variable_kernel_size "$v_kernel_size" \
      --des 'Exp' \
      --d_model 512 \
      --d_ff 512 \
      --itr 1 \
      --result_path 'result_long_term_forecast_iTransformer_dCNN.txt'
  done
done