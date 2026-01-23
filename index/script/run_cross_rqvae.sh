# nohup bash script/run_cross_rqvae.sh > /home/yuanmeng/xy_code/MQL4GRec/index/log/Instruments,Arts,Games,Pet,Cell,Automotive,Tools,Toys,Sports/nohup.log 2>&1 &
begin_cross_layer=0
Datasets='Instruments,Arts,Games,Pet,Cell,Automotive,Tools,Toys,Sports'
TIMESTAMP=$(date +%m%d-%H%M)
OUTPUT_DIR=log/$Datasets/cross_256/${TIMESTAMP}

mkdir -p $OUTPUT_DIR

python -u main.py \
  --num_emb_list 256 256 256 256 \
  --sk_epsilons 0.0 0.0 0.0 0.003 \
  --device cuda:0 \
  --datasets $Datasets \
  --ckpt_dir $OUTPUT_DIR \
  --data_root /home/yuanmeng/xy_code/datasets/amazon \
  --text_embedding_file .emb-llama-td.npy \
  --image_embedding_file .emb-ViT-L-14.npy \
  --eval_step 5 \
  --batch_size 4096 \
  --begin_cross_layer $begin_cross_layer \
  --use_cross_rq  \
  --text_contrast_weight 0.001 \
  --image_contrast_weight 0.001 \
  --epochs 500 > $OUTPUT_DIR/train.log
