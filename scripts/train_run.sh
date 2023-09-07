export OMP_NUM_THREADS=4

# train with split_v3 dataset
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=3 --node_rank=0 --master_addr="11.37.6.163" --master_port=11115 train_cross_domain_emb.py \
    --embedding_size 512 \
    --workers 10 \
    --epochs 10 \
    --decay_epochs 3 \
    --batch_size 80 \
    --lr 0.01 \
    --save_freq 50000 \
    --checkpoints './outputs/checkpoints/XXX' \
    --pretrained ./outputs/checkpoints/checkpoint_120.pth.tar \
    --mixed_precision_training \
    --finetune \
    --train_file ./datasets/train_file/new_train_98000.txt \
    --cls_file  ./datasets/train_file/cluster_10.txt\
    --cls_num 10 \
    --sample_info_file ./datasets/train_file/new_train_98000.txt\
    --goods_img_root  \
    --goods_text_root   \
    --photo_img_root  \
    --photo_text_root   \
    --clip_length 5 \
    >> ./outputs/logs/XXX/training.log
    # --resume './outputs/checkpoints/XXX/checkpoint_10.pth.tar' \


