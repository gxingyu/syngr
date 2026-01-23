import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import random

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer

class CrossRQVAE(nn.Module):
    def __init__(self,
                 text_in_dim=4096,
                 image_in_dim=768,
                 num_emb_list=[256,256,256,256],
                 e_dim=32,
                 layers=[2048,1024,512,256,128,64],
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons=[0.0,0.0,0.0,0.003],
                 sk_iters=100,
                 use_linear=0,
                 begin_cross_layer=0,
                 use_cross_rq=True,
                 text_contrast_weight=1.0,
                 image_contrast_weight=1.0,
                 text_lm_weight=1.0,
                 image_lm_weight=1.0,
                 recon_contrast_weight=0.001,
        ):
        super(CrossRQVAE, self).__init__()

        self.text_in_dim = text_in_dim
        self.image_in_dim = image_in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.num_rq_layers = len(num_emb_list)
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.begin_cross_layer = begin_cross_layer
        self.use_cross_rq = use_cross_rq
        
        self.align_dim = 768
        self.text_contrast_weight = text_contrast_weight
        self.image_contrast_weight = image_contrast_weight
        self.text_lm_weight = text_lm_weight
        self.image_lm_weight = image_lm_weight
        self.recon_contrast_weight = recon_contrast_weight
        self.text_encode_layer_dims = [self.align_dim] + self.layers + [self.e_dim]
        self.image_encode_layer_dims = [self.align_dim] + self.layers + [self.e_dim]
        # 先统一到一个维度
        self.text_align_encoder = MLPLayers(layers=[self.text_in_dim, self.align_dim],
                                            dropout=self.dropout_prob,bn=self.bn)
        self.image_align_encoder = MLPLayers(layers=[self.image_in_dim, self.align_dim],
                                            dropout=self.dropout_prob,bn=self.bn)
        self.text_align_decoder = MLPLayers(layers=[self.align_dim, self.text_in_dim],
                                            dropout=self.dropout_prob,bn=self.bn)
        self.image_align_decoder = MLPLayers(layers=[self.align_dim, self.image_in_dim],
                                            dropout=self.dropout_prob,bn=self.bn)

        # text
        self.text_encoder = MLPLayers(layers=self.text_encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)

        self.text_rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,
                                          use_linear=use_linear)

        self.text_decode_layer_dims = self.text_encode_layer_dims[::-1]
        self.text_decoder = MLPLayers(layers=self.text_decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)

        # image 
        self.image_encoder = MLPLayers(layers=self.image_encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)
        
        self.image_rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,
                                          use_linear=use_linear)

        self.image_decode_layer_dims = self.image_encode_layer_dims[::-1]
        self.image_decoder = MLPLayers(layers=self.image_decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)

    def class_con_cross_rq(
        self,
        text_vq,
        image_vq,
        residual_text_x,
        residual_image_x,
        text_x,
        image_x,
        use_sk=True,
        temperature=0.1
    ):
        # 1. 残差向量量化（保持不变）
        text_x_res, text_loss, text_indices, text_distances = text_vq(
            residual_text_x, use_sk=use_sk
        )
        image_x_res, image_loss, image_indices, image_distances = image_vq(
            residual_image_x, use_sk=use_sk
        )

        # 2. 跨模态对比学习（CLIP-style）
        batch_size = residual_text_x.size(0)

        # 使用量化前的残差特征（更稳定）
        text_feat = F.normalize(residual_text_x, dim=1)    # [B, D]
        image_feat = F.normalize(residual_image_x, dim=1)  # [B, D]

        # 相似度矩阵 [B, B]
        logits = torch.matmul(text_feat, image_feat.T) / temperature

        # 正样本：对角线
        labels = torch.arange(batch_size, device=logits.device)

        # text -> image
        loss_t2i = F.cross_entropy(logits, labels)

        # image -> text
        loss_i2t = F.cross_entropy(logits.T, labels)

        cross_modal_loss = 0.5 * (loss_t2i + loss_i2t)

        # 3. 将跨模态对比损失并入量化损失
        text_loss = text_loss + self.text_contrast_weight * cross_modal_loss
        image_loss = image_loss + self.image_contrast_weight * cross_modal_loss

        return (
            text_x_res,
            text_loss,
            text_indices,
            text_distances,
            image_x_res,
            image_loss,
            image_indices,
            image_distances,
        )

    def forward(self, text_x, image_x, use_sk=True):
        text_align_in = self.text_align_encoder(text_x)
        image_align_in = self.image_align_encoder(image_x)
        text_x = self.text_encoder(text_align_in)
        image_x = self.image_encoder(image_align_in)
        
        text_rq_loss = []
        image_rq_loss = []
        text_indices_list = []
        image_indices_list = []
        text_distances_list = []
        image_distances_list = []
        text_x_q = 0
        image_x_q = 0
        residual_text_x = text_x
        residual_image_x = image_x
        if self.use_cross_rq:
            for i in range(self.num_rq_layers):
                text_vq = self.text_rq.vq_layers[i]
                image_vq = self.image_rq.vq_layers[i]
                if i >= self.begin_cross_layer:
                    text_x_res, text_loss, text_indices, text_distances, image_x_res, image_loss, image_indices, image_distances = self.class_con_cross_rq(text_vq, image_vq, residual_text_x, residual_image_x, text_x, image_x, use_sk=use_sk)
                    residual_text_x = residual_text_x - text_x_res
                    residual_image_x = residual_image_x - image_x_res
                    text_x_q = text_x_q + text_x_res
                    image_x_q = image_x_q + image_x_res
                else:
                    text_x_res, text_loss, text_indices, text_distances = text_vq(residual_text_x, use_sk=use_sk)
                    image_x_res, image_loss, image_indices, image_distances = image_vq(residual_image_x, use_sk=use_sk)
                    residual_text_x = residual_text_x - text_x_res
                    residual_image_x = residual_image_x - image_x_res
                    text_x_q = text_x_q + text_x_res
                    image_x_q = image_x_q + image_x_res
                text_rq_loss.append(text_loss)
                text_indices_list.append(text_indices)
                text_min_distance = text_distances.min(dim=-1)[0] 
                text_distances_list.append(text_min_distance)

                image_rq_loss.append(image_loss)    
                image_indices_list.append(image_indices)
                image_min_distance = image_distances.min(dim=-1)[0] 
                image_distances_list.append(image_min_distance)
            text_rq_loss = torch.stack(text_rq_loss).mean()
            image_rq_loss = torch.stack(image_rq_loss).mean()
            text_indices = torch.stack(text_indices_list, dim=-1)
            image_indices = torch.stack(image_indices_list, dim=-1)
            text_distances = torch.stack(text_distances_list, dim=1)
            image_distances = torch.stack(image_distances_list, dim=1)
        else:
            text_x_q, text_rq_loss, text_indices, text_distances = self.text_rq(text_x, use_sk=use_sk)
            image_x_q, image_rq_loss, image_indices, image_distances = self.image_rq(image_x, use_sk=use_sk)
        lm_loss = torch.nn.functional.mse_loss(text_x_q, image_x_q)
        text_rq_loss = text_rq_loss + self.text_lm_weight * lm_loss
        image_rq_loss = image_rq_loss + self.image_lm_weight * lm_loss
        text_align_out = self.text_decoder(text_x_q)
        image_align_out = self.image_decoder(image_x_q)
        text_out = self.text_align_decoder(text_align_out)
        image_out = self.image_align_decoder(image_align_out)
        share_out = (text_x_q, image_x_q)

        return text_out, image_out, text_rq_loss, image_rq_loss, text_indices, image_indices, text_distances, image_distances, share_out


    def text_image_recon_align(self, text_out, image_out, temperature=0.1):

        text_out_norm = F.normalize(text_out, p=2, dim=1)
        image_out_norm = F.normalize(image_out, p=2, dim=1)
        sim_matrix = torch.matmul(text_out_norm, image_out_norm.T) / temperature  # [batch, batch]
        batch_size = text_out.size(0)
        labels = torch.arange(batch_size, device=text_out.device)
        loss = F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)
        return loss
    
    def compute_loss(self, text_out, image_out, text_rq_loss, image_rq_loss, text_indices, image_indices, text_distances, image_distances, text_xs, image_xs, share_out):
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(text_out, text_xs, reduction='mean') + F.mse_loss(image_out, image_xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(text_out, text_xs, reduction='mean') + F.l1_loss(image_out, image_xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')
        align_loss = self.text_image_recon_align(share_out[0], share_out[1])
        loss_total = loss_recon + self.quant_loss_weight * (text_rq_loss + image_rq_loss) + self.recon_contrast_weight * align_loss
        return loss_total, loss_recon
    
    @torch.no_grad()
    def get_indices(self, text_xs, image_xs, use_sk=False, text=0):
        text_align_in = self.text_align_encoder(text_xs)
        image_align_in = self.image_align_encoder(image_xs)
        text_x_e = self.text_encoder(text_align_in)
        image_x_e = self.image_encoder(image_align_in)
        residual_text_x = text_x_e
        residual_image_x = image_x_e
        _, _, text_indices, text_distances = self.text_rq(text_x_e, use_sk=use_sk)
        _, _, image_indices, image_distances = self.image_rq(image_x_e, use_sk=use_sk)
        if text == 0:
            return text_indices, image_indices, text_distances, image_distances
        elif text == 1:
            return text_indices,text_distances
        else:
            return image_indices, image_distances