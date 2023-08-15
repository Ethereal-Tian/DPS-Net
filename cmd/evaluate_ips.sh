
eval_datadir='/home/chaoran/data'
eval_dataset='IPS'
reflect_type='synthetic'

restore_ckpt='checkpoints/pretrain/dps-net_IPS_99.pth'

CUDA_VISIBLE_DEVICES=3 python eval.py --batch_size 1 \
--spatial_scale -0.2 0.4 --saturation_range 0 1.4  --num_steps 200000 --mixed_precision \
--train_gru_iters 8 --valid_gru_iters 8 \
--datadir ${eval_datadir} --dataset ${eval_dataset} --reflect_type ${reflect_type} \
--training_mode eval --restore_ckpt ${restore_ckpt}
