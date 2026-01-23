# Arts Games Instruments
export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1

Index_file=.index_lemb.json
Image_index_file=.index_vitemb.json

Datasets='Arts'
data_path=../datasets/amazon/


OUTPUT_DIR=./log/$Datasets/20251221_152146

num_beams=20


python ensemble.py \
    --output_dir $OUTPUT_DIR\
    --dataset $Datasets\
    --data_path $data_path\
    --index_file $Index_file\
    --image_index_file $Image_index_file\
    --num_beams $num_beams