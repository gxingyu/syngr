Datasets='Instruments,Arts,Games,Pet,Cell,Automotive,Tools,Toys,Sports'


python -u generate_indices_distance_mul.py \
  --datasets $Datasets \
  --data_root '/home/yuanmeng/xy_code/datasets/amazon' \
  --text_embedding_file .emb-llama-td.npy \
  --image_embedding_file .emb-ViT-L-14.npy \
  --device cuda:0 \
  --ckpt_path log/$Datasets/cross_256/1220-2052/best_text_collision_model.pth \
  --output_file '.index_lemb.json' \
  --content text

python -u generate_indices_distance_mul.py \
    --datasets $Datasets \
    --data_root '/home/yuanmeng/xy_code/datasets/amazon' \
    --text_embedding_file .emb-llama-td.npy \
    --image_embedding_file .emb-ViT-L-14.npy \
    --device cuda:0 \
    --ckpt_path log/$Datasets/cross_256/1220-2052/best_image_collision_model.pth \
    --output_file '.index_vitemb.json' \
    --content image

