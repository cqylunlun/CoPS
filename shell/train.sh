gpu_id=$1
e=$2
train_path=$3
train_name=${train_path,,}

clear
cd ..
CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
  --train_data_path /root/cqy/dataset/$train_path \
  --save_path ./results/models/$train_name \
  --dataset $train_name \
  --depth 8 \
  --n_ctx 12 \
  --t_n_ctx 4 \
  --features_list 24 \
  --epoch $e \
  --batch_size 8 \
  --image_size 518 \
  --dpam 24