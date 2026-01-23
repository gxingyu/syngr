# setsid bash script/finetune.sh > /dev/null 2>&1 &
# Arts Games Instruments
export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,2

Index_file=.index_lemb_256_dis_all_right.json
Image_index_file=.index_vitemb_256_dis_all_right.json

# Tasks='seqrec,seqimage,item2image,image2item,seqitem2image,seqimage2item'
Tasks='seqrec,seqimage,seqitem2image,seqimage2item,item2image,image2item'
Valid_task=seqrec
Per_device_batch_size=2048
Datasets='Arts'
# Datasets='Games'
# Datasets='Instruments'
data_path=/home/yuanmeng/xy_code/datasets/amazon
load_model_name=/home/yuanmeng/xy_code/MQL4GRec/log/Pet,Cell,Automotive,Tools,Toys,Sports/ckpt_b2048_lr1e-3_seqrec,seqimage_30/pretrain
timestamp=$(date "+%Y%m%d_%H%M%S")
OUTPUT_DIR=./log/$Datasets/$Tasks
mkdir -p $OUTPUT_DIR
log_file=$OUTPUT_DIR/train.log

torchrun --nproc_per_node=2 --master_port=2309 finetune_mask.py \
    --data_path $data_path \
    --dataset $Datasets \
    --output_dir $OUTPUT_DIR \
    --load_model_name $load_model_name \
    --per_device_batch_size $Per_device_batch_size \
    --learning_rate 5e-4 \
    --epochs 200 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --logging_step 50 \
    --max_his_len 20 \
    --prompt_num 4 \
    --patient 10 \
    --index_file $Index_file \
    --image_index_file $Image_index_file \
    --tasks $Tasks \
    --fp16 \
    --alpha 0.1 \
    --mask_ratio 0.3 \
    --temperature 0.07 \
    --valid_task $Valid_task > $log_file

num_beams=20
Valid_task=seqrec
results_file=$OUTPUT_DIR/results_${Valid_task}_${num_beams}.json
save_file=$OUTPUT_DIR/save_${Valid_task}_${num_beams}.json

torchrun --nproc_per_node=2 --master_port=2309 test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR \
    --data_path $data_path \
    --dataset $Datasets \
    --test_batch_size 256 \
    --num_beams $num_beams \
    --index_file $Index_file \
    --image_index_file $Image_index_file \
    --test_task $Valid_task \
    --results_file $results_file \
    --save_file $save_file \
    --filter_items
Valid_task=seqimage
results_file=$OUTPUT_DIR/results_${Valid_task}_${num_beams}.json
save_file=$OUTPUT_DIR/save_${Valid_task}_${num_beams}.json
torchrun --nproc_per_node=2 --master_port=2309 test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR \
    --data_path $data_path \
    --dataset $Datasets \
    --test_batch_size 256 \
    --num_beams $num_beams \
    --index_file $Index_file \
    --image_index_file $Image_index_file \
    --test_task $Valid_task \
    --results_file $results_file \
    --save_file $save_file \
    --filter_items

python ensemble.py \
    --output_dir $OUTPUT_DIR\
    --dataset $Datasets\
    --data_path $data_path\
    --index_file $Index_file\
    --image_index_file $Image_index_file\
    --num_beams $num_beams

