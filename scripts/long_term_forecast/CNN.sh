export CUDA_VISIBLE_DEVICES=0

model_name=CNN

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id cnn \
  --model $model_name \
  --data custom \
  --features 'MS' \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 21 \
  --c_out 7 \
  --des "Exp" \
  --itr 1