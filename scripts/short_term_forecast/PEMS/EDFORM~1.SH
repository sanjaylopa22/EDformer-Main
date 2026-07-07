#export CUDA_VISIBLE_DEVICES=0

model_name=EDformer_frequency

seq_len=96
pred_len=24

down_sampling_layers=3
down_sampling_window=2
learning_rate=0.003
d_model=128
batch_size=32
train_epochs=10
patience=3

python -u run.py \
 --task_name long_term_forecast \
 --is_training 1 \
 --root_path ./dataset/PEMS03/ \
 --data_path PEMS03.npz \
 --model_id PEMS03 \
 --model $model_name \
 --data PEMS \
 --features M \
 --seq_len 96 \
 --label_len 0 \
 --pred_len 48 \
 --enc_in 358 \
 --dec_in 358 \
 --c_out 358 \
 --channel_independence 0 \
 --d_model 128 \
 --batch_size 32 \
 --learning_rate 0.003 \
 --train_epochs 10 \
 --patience 3 \
 --down_sampling_layers 3 \
 --down_sampling_method avg \
 --down_sampling_window 2\
 --des 'Exp_learnable_alpha' \
 --itr 1 \
 --alpha 0.1 \
 --learnable_alpha 1 \
 --taper_ratio 0.1


python -u run.py \
 --task_name long_term_forecast \
 --is_training 1 \
 --root_path ./dataset/PEMS04/ \
 --data_path PEMS04.npz \
 --model_id PEMS04 \
 --model $model_name  \
 --data PEMS \
 --features M \
 --seq_len 96 \
 --label_len 0 \
 --pred_len 48\
 --enc_in 307 \
 --dec_in 307 \
 --c_out 307 \
 --channel_independence 0 \
 --d_model 128 \
 --batch_size 32 \
 --learning_rate 0.003 \
 --train_epochs 10 \
 --patience 3 \
 --down_sampling_layers 3 \
 --down_sampling_method avg \
 --down_sampling_window 2\
 --des 'Exp_learnable_alpha' \
 --itr 1 \
 --alpha 0.1 \
 --learnable_alpha 1 \
 --taper_ratio 0.1


python -u run.py \
 --task_name long_term_forecast \
 --is_training 1 \
 --root_path ./dataset/PEMS07/ \
 --data_path PEMS07.npz \
 --model_id PEMS07 \
 --model $model_name  \
 --data PEMS \
 --features M \
 --seq_len 96 \
 --label_len 0 \
 --pred_len 48 \
 --enc_in 883 \
 --dec_in 883 \
 --c_out 883 \
  --des 'Exp_learnable_alpha' \
 --itr 1 \
 --alpha 0.1 \
 --learnable_alpha 1 \
 --taper_ratio 0.1\
 --channel_independence 0 \
 --d_model 128 \
 --batch_size 32 \
 --learning_rate 0.003 \
 --train_epochs 10 \
 --patience 3 \
 --down_sampling_layers 3 \
 --down_sampling_method avg \
 --down_sampling_window 2


python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/PEMS08/ \
  --data_path PEMS08.npz \
  --model_id PEMS08 \
 --model $model_name  \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 48 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --des 'Exp_learnable_alpha' \
 --itr 1 \
 --alpha 0.1 \
 --learnable_alpha 1 \
 --taper_ratio 0.1\
 --channel_independence 0 \
 --d_model 128 \
 --batch_size 32 \
 --learning_rate 0.003 \
 --train_epochs 10 \
 --patience 3 \
 --down_sampling_layers 3 \
 --down_sampling_method avg \
 --down_sampling_window 2
