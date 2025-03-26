export CUDA_VISIBLE_DEVICES=0

model_name=CNN

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/mine/ \
  --data_path linear_1.csv \
  --model_id linear_1_cnn \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --c_out 7 \
  --des 'Mine' \
  --d_model 64 \
  --d_ff 128 \
  --batch_size 4 \
  --itr 1