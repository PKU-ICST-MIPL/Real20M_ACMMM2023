export OMP_NUM_THREADS=4

model_path='outputs/checkpoints/XXX/checkpoint_10.pth.tar'
version=XXX
output_path=XXX/evaluate/i2i/feat/$version

if [ -e $output_path ]
then
    rm -rf $output_path
fi

mkdir -p $output_path/goods
mkdir -p $output_path/video

# better to set the absolute path for pretrained pth
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=52293 train_cross_domain_emb.py \
    --embedding_size 512 \
    --workers 10 \
    --epochs 10000 \
    --decay_epochs 30 \
    --batch_size 80 \
    --lr 0.005 \
    --save_freq 50000 \
    --checkpoints './outputs/checkpoints/product_cross_domain_emb_v1.0' \
    --pretrained './outputs/checkpoints/checkpoint_120.pth.tar' \
    --finetune \
    --mixed_precision_training \
    --goods_img_root  \
    --goods_text_root \
    --photo_img_root \
    --photo_text_root  \
    --clip_length 5 \
    --test_file './datasets/test_file/new_goods2video_goods.txt' \
    --evaluate \
    --resume $model_path \
    --output_dir $output_path/goods

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=52293 train_cross_domain_emb.py \
    --embedding_size 512 \
    --workers 10 \
    --epochs 10000 \
    --decay_epochs 30 \
    --batch_size 80 \
    --lr 0.005 \
    --save_freq 50000 \
    --checkpoints './outputs/checkpoints/product_cross_domain_emb_v1.0' \
    --pretrained './outputs/checkpoints/checkpoint_120.pth.tar' \
    --finetune \
    --mixed_precision_training \
    --goods_img_root '/images' \
    --goods_text_root '/texts'  \
    --photo_img_root '/images' \
    --photo_text_root '/texts'  \
    --clip_length 5 \
    --test_file './datasets/test_file/all_videos.txt' \
    --evaluate \
    --resume $model_path \
    --output_dir $output_path/video

cat $output_path/goods/* > $output_path/goods/query.feat
cat $output_path/video/* > $output_path/video/doc.feat

cd benchmark/evaluate/goods2video/index
sh run.sh $output_path $version

