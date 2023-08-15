train_datadir='/home/chaoran/data'

train_dataset='IPS'
reflect_type='synthetic'
end_epoch=100
restore_ckpt="null"

batch_size=8

CUDA_VISIBLE_DEVICES=3 python train.py --batch_size ${batch_size} \
 --spatial_scale -0.2 0.4 --saturation_range 0 1.4  --num_steps 200000 --mixed_precision \
 --train_gru_iters 8  --valid_gru_iters 8\
 --start_epoch 0 --end_epoch ${end_epoch} \
 --datadir ${train_datadir} --dataset ${train_dataset} --reflect_type ${reflect_type} \
 --restore_ckpt ${restore_ckpt}