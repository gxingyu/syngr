import argparse
import random
import torch
import numpy as np
from time import time
import logging
import json
from torch.utils.data import DataLoader

from datasets import EmbDataset, DualEmbDataset, DualEmbAllDataset
from models.cross_rqvae import CrossRQVAE
from trainer import  CrossTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, )
    parser.add_argument('--eval_step', type=int, default=1, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument('--text_embedding_file', type=str, default='Scientific')
    parser.add_argument('--image_embedding_file', type=str, default='Scientific')
    parser.add_argument('--data_root', type=str, default='Scientific')
    parser.add_argument('--datasets', type=str, default='Scientific')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")
    parser.add_argument("--use_cross_rq", action="store_true", help="use cross rq or not")
    parser.add_argument("--bn", action="store_true", help="use batch norm or not")
    parser.add_argument("--begin_cross_layer", type=int, default=4, help="begin cross layer")

    parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256,256,256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument('--layers', type=int, nargs='+', default=[2048,1024,512,256,128,64], help='hidden sizes of every layer')

    parser.add_argument("--ckpt_dir", type=str, default="./log", help="output directory for model")

 
    parser.add_argument("--text_contrast_weight", type=float, default=1.0, help="text contrast weight")
    parser.add_argument("--image_contrast_weight", type=float, default=1.0, help="image contrast weight")
    parser.add_argument("--recon_contrast_weight", type=float, default=0.001, help="recon contrast weight")

    return parser.parse_args()




if __name__ == '__main__':
    """fix the random seed"""
    seed = 2023
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    print(args)
    
    logging.basicConfig(level=logging.DEBUG)

    """build dataset"""
    print("use cross rq", args.use_cross_rq)
    print("begin cross layer", args.begin_cross_layer)
    data = DualEmbAllDataset(args)
    model = CrossRQVAE(text_in_dim=data.text_dim,
                       image_in_dim=data.img_dim,
                        num_emb_list=args.num_emb_list,
                        e_dim=args.e_dim,
                        layers=args.layers,
                        dropout_prob=args.dropout_prob,
                        bn=args.bn,
                        loss_type=args.loss_type,
                        quant_loss_weight=args.quant_loss_weight,
                        kmeans_init=args.kmeans_init,
                        kmeans_iters=args.kmeans_iters,
                        sk_epsilons=args.sk_epsilons,
                        sk_iters=args.sk_iters,
                        use_cross_rq=args.use_cross_rq,
                        begin_cross_layer=args.begin_cross_layer,
                    )
    print(model)
    data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)
    trainer = CrossTrainer(args,model)
    best_loss, best_text_collision_rate, best_image_collision_rate = trainer.fit(data_loader)

    print("Best Loss",best_loss)
    print("Best Text Collision Rate", best_text_collision_rate)
    print("Best Image Collision Rate", best_image_collision_rate) 

