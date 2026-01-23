cd index
bash script/run_cross_rqvae.sh
bash script/gen_code_dis.sh
cd ..
bash script/pretrain.sh
bash script/finetune.sh
bash script/test_ddp_save.sh


