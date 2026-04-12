gpu_id=$1
e=$2
train_path=$3
test_path=$4
train_name=${train_path,,}
test_name=${test_path,,}

clear
cd ..
CUDA_VISIBLE_DEVICES=$gpu_id python test.py \
  --data_path /root/cqy/dataset/$test_path \
  --checkpoint_path ./results/models/$train_name/epoch_$e.pth \
  --dataset $test_name \
  --depth 8 \
  --n_ctx 12 \
  --t_n_ctx 4 \
  --features_list 24 \
  --batch_size 1 \
  --image_size 518 \
  --dpam 24