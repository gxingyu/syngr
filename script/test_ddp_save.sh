# Arts Games Instruments
export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1

CUDA_CUDA_VISIBLE_DEVICES=0,2
Index_file=.index_lemb.json
Image_index_file=.index_vitemb.json

Datasets='Instruments'
data_path=../datasets/amazon/

OUTPUT_DIR=./log/$Datasets/20251224_225403

num_beams=20
Valid_task=seqrec
results_file=$OUTPUT_DIR/results_${Valid_task}_${num_beams}.json
save_file=$OUTPUT_DIR/save_${Valid_task}_${num_beams}.json

torchrun --nproc_per_node=1 --master_port=2309 test_ddp_save.py \
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
torchrun --nproc_per_node=1 --master_port=2309 test_ddp_save.py \
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