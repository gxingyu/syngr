import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from typing import Optional, Tuple, Union

class SynergisticT5(T5ForConditionalGeneration):
    def __init__(self, config, alpha=0.1, mask_ratio=0.3, temperature=0.07, prompt_len=0):
        super().__init__(config)
        self.alpha = alpha
        self.mask_ratio = mask_ratio
        self.temperature = temperature
        
        self.prompt_len = prompt_len
        
    def _get_dominant_mask(self, input_ids, attention_mask):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True
        )
        
        last_layer_attn = encoder_outputs.attentions[-1] 
        token_importance = last_layer_attn.mean(dim=1).mean(dim=1) 
        
        batch_size, seq_len = input_ids.shape
        masked_input_ids = input_ids.clone()
        

        ITEM_STRIDE = 8 
        TEXT_LEN = 4
        

        start_index = self.prompt_len
        
        for b in range(batch_size):
            text_scores = []
            img_scores = []
            

            for i in range(start_index, seq_len, ITEM_STRIDE):

                if i + ITEM_STRIDE > seq_len:
                    break
                    

                if input_ids[b, i] == self.config.pad_token_id:
                    break
                

                t_score = token_importance[b, i : i+TEXT_LEN].mean()
                i_score = token_importance[b, i+TEXT_LEN : i+ITEM_STRIDE].mean()
                
                text_scores.append(t_score)
                img_scores.append(i_score)
            
            if not text_scores: continue
            
            avg_text_score = sum(text_scores) / len(text_scores)
            avg_img_score = sum(img_scores) / len(img_scores)
            
            is_text_dominant = avg_text_score > avg_img_score
            

            dominant_indices = []
            

            for i in range(start_index, seq_len, ITEM_STRIDE):
                if i + ITEM_STRIDE > seq_len: break
                if input_ids[b, i] == self.config.pad_token_id: break
                
                if is_text_dominant:
                    dominant_indices.extend(range(i, i + TEXT_LEN))
                else:
                    dominant_indices.extend(range(i + TEXT_LEN, i + ITEM_STRIDE))
            
            if dominant_indices:
                dom_scores = token_importance[b, dominant_indices]
                k = int(len(dominant_indices) * self.mask_ratio)
                if k > 0:
                    _, top_k_indices = torch.topk(dom_scores, k)
                    global_indices_to_mask = [dominant_indices[idx] for idx in top_k_indices]
                    masked_input_ids[b, global_indices_to_mask] = self.config.pad_token_id 
                    
        return masked_input_ids

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        unimodal_input_ids: Optional[torch.LongTensor] = None,
        **kwargs
    ):

        if unimodal_input_ids is None or not self.training:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
            

        with torch.no_grad():
            masked_input_ids = self._get_dominant_mask(input_ids, attention_mask)
            

        if unimodal_input_ids.shape[1] < input_ids.shape[1]:
            padding = torch.full((unimodal_input_ids.shape[0], input_ids.shape[1] - unimodal_input_ids.shape[1]), 
                                 self.config.pad_token_id, device=unimodal_input_ids.device)
            unimodal_input_ids = torch.cat([unimodal_input_ids, padding], dim=1)
        elif unimodal_input_ids.shape[1] > input_ids.shape[1]:
            unimodal_input_ids = unimodal_input_ids[:, :input_ids.shape[1]]

        concat_input_ids = torch.cat([input_ids, masked_input_ids, unimodal_input_ids], dim=0)
        

        concat_attention_mask = (concat_input_ids != self.config.pad_token_id).long()
        

        concat_labels = labels.repeat(3, 1)


        outputs = super().forward(
            input_ids=concat_input_ids,
            attention_mask=concat_attention_mask,
            labels=concat_labels,
            output_hidden_states=True,
            **kwargs
        )
        

        batch_size = input_ids.shape[0]

        hidden_states = outputs.decoder_hidden_states[-1]
        

        seq_repr = hidden_states.mean(dim=1) 
        
        z_orig = seq_repr[:batch_size]
        z_mask = seq_repr[batch_size : 2*batch_size]
        z_uni  = seq_repr[2*batch_size :]
        

        sim_pos = F.cosine_similarity(z_mask, z_orig, dim=-1) / self.temperature

        sim_neg = F.cosine_similarity(z_mask, z_uni, dim=-1) / self.temperature
        

        
        contrastive_logits = torch.stack([sim_pos, sim_neg], dim=1)

        contrastive_labels = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)
        
        loss_cl = F.cross_entropy(contrastive_logits, contrastive_labels)
        

        total_loss = outputs.loss + self.alpha * loss_cl
        

        outputs.loss = total_loss
        
        return outputs