train_datadir='/mnt/nas_8/datasets/tiancr/DPS-Net-Data'

train_dataset='RPS'
reflect_type='real'
end_epoch=150
restore_ckpt='checkpoints/pretrain/dps-net_IPS_99.pth'

batch_size=8

CUDA_VISIBLE_DEVICES=3 python train.py --batch_size ${batch_size} \
 --spatial_scale -0.2 0.4 --saturation_range 0 1.4  --num_steps 200000 --mixed_precision \
 --train_gru_iters 8  --valid_gru_iters 8\
 --start_epoch 0 --end_epoch ${end_epoch} \
 --datadir ${train_datadir} --dataset ${train_dataset} --reflect_type ${reflect_type} \
 --restore_ckpt ${restore_ckpt}