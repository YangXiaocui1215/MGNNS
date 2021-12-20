import torch
import torch.nn as nn
import torch.nn.functional as F
from models.multi_head_att.submodules import MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding
from models.multi_head_att.submodules import  ScaledDotProductAttention, LayerNorm
import numpy as np


def masked_mean(input, mask=None, dim=1):
    # input: [batch_size, seq_len, hidden_size]
    # mask: Float Tensor of size [batch_size, seq_len], where 1.0 for unmask, 0.0 for mask ones
    if mask is None:
        return torch.mean(input, dim=dim)
    else:
        mask = mask.unsqueeze(-1)
        mask_input = input * mask
        sum_mask_input = mask_input.sum(dim=dim)
        for dim in range(mask.size(0)):
            sum_mask_input[dim] = sum_mask_input[dim] / mask[dim].sum()
        return sum_mask_input


def masked_max(input, mask=None, dim=1):
    # input: [batch_size, seq_len, hidden_size]
    # mask: Float Tensor of size [batch_size, seq_len], where 1.0 for unmask, 0.0 for mask ones
    if mask is None:
        max_v, _ = torch.max(input, dim=dim)
        return max_v
    else:
        mask = mask.unsqueeze(-1)
        mask = mask.repeat(1, 1, input.size(-1))
        input = input.masked_fill(mask == 0.0, float('-inf'))
        max_v, _ = torch.max(input, dim=dim)
        return max_v


class MaskedSoftmax(nn.Module):
    def __init__(self, dim):
        super(MaskedSoftmax, self).__init__()
        self.dim = dim

    def forward(self, logit, mask=None):
        if mask is None:
            dist = F.softmax(logit - torch.max(logit, dim=self.dim, keepdim=True)[0], dim=self.dim)
        else:
            dist_ = F.softmax(logit - torch.max(logit, dim=self.dim, keepdim=True)[0], dim=self.dim) * mask
            normalization_factor = dist_.sum(self.dim, keepdim=True)
            dist = dist_ / normalization_factor
        return dist

class CoAttention(nn.Module):
    def __init__(self, text_feat_size, img_object_feat_size, img_place_feat_size, interaction_type='co_att'):
        """Initialize model."""
        super(CoAttention, self).__init__()
        print('-----------------------------Multi-modal features interaction by CoAttention-------------------')
        self.text_feat_size = text_feat_size
        self.img_object_feat_size = img_object_feat_size
        self.img_place_feat_size = img_place_feat_size
        self.interaction_type = interaction_type

        self.v_text_object = nn.Linear(text_feat_size, 1, bias=False)
        self.v_text_place = nn.Linear(text_feat_size, 1, bias=False)
        self.v_img_object = nn.Linear(img_object_feat_size, 1, bias=False)
        self.v_img_place = nn.Linear(img_place_feat_size, 1, bias=False)

        self.text2img_object_project = nn.Linear(text_feat_size, img_object_feat_size, bias=False)
        self.text2img_place_project = nn.Linear(text_feat_size, img_place_feat_size, bias=False)
        self.img_object2text_project = nn.Linear(img_object_feat_size, text_feat_size, bias=False)
        self.img_place2text_project = nn.Linear(img_place_feat_size, text_feat_size, bias=False)

        self.img_object_project = nn.Linear(img_object_feat_size, img_object_feat_size)
        self.img_place_project = nn.Linear(img_place_feat_size, img_place_feat_size)
        self.text_object_project = nn.Linear(text_feat_size, text_feat_size)
        self.text_place_project = nn.Linear(text_feat_size, text_feat_size)

        self.dropout = nn.Dropout(0.5)
        self.softmax = MaskedSoftmax(dim=1)
        ###text_feature 与 img_object_feature, img_palce_feature两两交互，故最后有四种特征， 最后映射的特征维度待定
        self.linear = nn.Linear(text_feat_size*2 + img_object_feat_size + img_place_feat_size, text_feat_size)

    def text_att_scores(self, text_feat, img_feats, img_type):
        batch_size, img_num, img_feat_size = list(img_feats.size())
        # print('------------------- batch_size, img_num, img_feat_size ----------')
        # print(batch_size, img_num, img_feat_size)
        batch_size, text_feat_size = list(text_feat.size())
        # print('----------------------------batch_size, text_feat_size--------------')
        # print(batch_size, text_feat_size)
        img_feats_ = img_feats.view(-1, img_feat_size)  # [batch_size*img_num, img_feat_size]
        if img_type=='object':
            img_feature = self.img_object2text_project(img_feats_) # [batch_size*img_num, text_feat_size]
             # Project decoder state: text_feats (in our case)
            text_feature = self.text_object_project(text_feat)  # [batch_size, text_feat_size]
            text_feature_expanded = text_feature.unsqueeze(1).expand(batch_size, img_num, text_feat_size).contiguous()
            text_feature_expanded = text_feature_expanded.view(-1, text_feat_size)  # [batch_size*img_num, text_feat_size]
        elif img_type=='place':
            img_feature = self.img_place2text_project(img_feats_) # [batch_size*img_num, text_feat_size]
            # Project decoder state: text_feats (in our case)
            text_feature = self.text_place_project(text_feat)  # [batch_size, text_feat_size]
            text_feature_expanded = text_feature.unsqueeze(1).expand(batch_size, img_num, text_feat_size).contiguous()
            text_feature_expanded = text_feature_expanded.view(-1, text_feat_size)  # [batch_size*img_num, text_feat_size]

        # sum up attention features
        att_features = img_feature + text_feature_expanded  # [batch_size*img_num, text_feat_size]
        e = torch.tanh(att_features)  # [batch_size*img_num, text_feat_size]
        if img_type == 'object':
            scores = self.v_text_object(e)  # [batch_size*img_num, 1]
        elif img_type == 'place':
            scores = self.v_text_place(e)  # [batch_size*img_num, 1]
        scores = scores.view(-1, img_num)  # [batch_size, img_num]
        return scores

    def img_att_scores(self, img_feat, text_feats, img_type):
        batch_size, max_src_len, text_feat_size = list(text_feats.size())
        batch_size, img_feat_size = list(img_feat.size())

        text_feats_ = text_feats.view(-1, text_feat_size)  # [batch_size*max_src_len, text_feat_size]
        if img_type=='object':
            text_feature = self.text2img_object_project(text_feats_) # [batch_size*max_src_len, img_feat_size]
            # Project decoder state: text_feats (in our case)
            img_feature = self.img_object_project(img_feat)  # [batch_size, img_object_feat_size]
            img_feature_expanded = img_feature.unsqueeze(1).expand(batch_size, max_src_len, img_feat_size).contiguous()
            img_feature_expanded = img_feature_expanded.view(-1, img_feat_size)  # [batch_size*max_src_len, img_feat_size]

        elif img_type=='place':
            text_feature = self.text2img_place_project(text_feats_) # [batch_size*max_src_len, img_feat_size]
              # Project decoder state: text_feats (in our case)
            img_feature = self.img_place_project(img_feat)  # [batch_size, img_place_feat_size]
            img_feature_expanded = img_feature.unsqueeze(1).expand(batch_size, max_src_len, img_feat_size).contiguous()
            img_feature_expanded = img_feature_expanded.view(-1, img_feat_size)  # [batch_size*max_src_len, img_feat_size]

        # sum up attention features
        att_features = text_feature + img_feature_expanded  # [batch_size*max_src_len, img_feat_size]
        e = torch.tanh(att_features)  # [batch_size*max_src_len, img_feat_size]
        if img_type=='object':
            scores = self.v_img_object(e) # [batch_size*max_src_len, 1]
        elif img_type=='place':
            scores = self.v_img_place(e) # [batch_size*max_src_len, 1]
        scores = scores.view(-1, max_src_len)  # [batch_size, max_src_len]
        return scores

    def forward(self, 
                text_feat, text_feats, 
                img_object_feat, img_object_feats, 
                img_place_feat, img_place_feats, 
                src_mask):
        '''
        text_feat:[batch_size, text_feat_size], 二维特征，输入self.text_att_scores(), 可将text_gcn输出特征视为text_feat
        text_feats:[batch_size, text_len, text_feat_size],三维特征，输入self.img_att_scores()
        img_object_feat: [batch_size, img_obj_feat_size], 二维特征，输入self.img_att_scores(), 可将image_gcn_att输出特征视为img_object_feat
        img_object_feats: [bacth_size, img_num, image_obj_feat_size], 三维特征，输入self.text_att_scores()
        img_place_feat: [batch_size, img_pla_feat_size], 二维特征，输入self.img_att_scores(), 可将image_gcn_att输出特征视为img_place_feat
        img_place_feats: [bacth_size, img_num, image_pla_feat_size], 三维特征，输入self.text_att_scores()
        '''
        # Text
        img_object_batch_size, img_object_num, img_object_feat_size = list(img_object_feats.size())
        img_place_batch_size, img_place_num, img_place_feat_size = list(img_place_feats.size())
        text_batch_size, text_max_len, text_feat_size = list(text_feats.size())

        '''-----------------------text_img_object_score, 寻找与文本特征相关的img_object_feats----------------'''
        text_img_object_scores = self.text_att_scores(text_feat, img_object_feats, img_type='object') # [batch_size, img_object_num]
        text_img_object_att_dist = self.softmax(text_img_object_scores)
        text_img_object_att_dist = text_img_object_att_dist.unsqueeze(1) # [batch_size, 1, img_object_num]
        img_object_feats = img_object_feats.view(-1, img_object_num, img_object_feat_size)  # [batch_size, img_object_num, img_object_feat_size]
        img_object_context = torch.bmm(text_img_object_att_dist, img_object_feats)  # [batch_size, 1, img_object_feat_size]
        img_object_context = img_object_context.squeeze(1)  # [batch_size, img_object_feat_size]

        '''-----------------------text_img_place_score, 寻找与文本特征相关的img_place_feats----------------'''
        text_img_place_scores = self.text_att_scores(text_feat, img_place_feats, img_type='place') # [batch_size, img_object_num]
        text_img_place_att_dist = self.softmax(text_img_place_scores)
        text_img_place_att_dist = text_img_place_att_dist.unsqueeze(1) # [batch_size, 1, img_place_num]
        img_place_feats = img_place_feats.view(-1, img_place_num, img_place_feat_size)  # [batch_size, img_place_num, img_place_feat_size]
        img_place_context = torch.bmm(text_img_place_att_dist, img_place_feats)  # [batch_size, 1, img_place_feat_size]
        img_place_context = img_place_context.squeeze(1)  # [batch_size, img_place_feat_size]

        '''-----------------------img_object_text_score, 寻找与img_object_feat相关的text_feats----------------'''
        ###text_feats是padding之后的
        img_object_text_scores = self.img_att_scores(img_object_feat, text_feats, img_type='object') #  [batch_size, max_text_len]
        img_object_text_att_dist = self.softmax(img_object_text_scores, mask=src_mask)
        img_object_text_att_dist = img_object_text_att_dist.unsqueeze(1)  # [batch_size, 1, max_text_len]
        text_feats = text_feats.view(-1, text_max_len, text_feat_size)  # [batch_size, text_max_len, text_feat_size]
        text_object_context = torch.bmm(img_object_text_att_dist, text_feats)  # [batch_size, 1, text_feat_size]
        text_object_context = text_object_context.squeeze(1)  # [batch_size, text_feat_size]
        '''-----------------------img_object_text_score, 寻找与img_object_feat相关的text_feats----------------'''
        ###text_feats是padding之后的
        img_place_text_scores = self.img_att_scores(img_place_feat, text_feats, img_type='place') #  [batch_size, max_text_len]
        img_place_text_att_dist = self.softmax(img_place_text_scores, mask=src_mask)
        img_place_text_att_dist = img_place_text_att_dist.unsqueeze(1)  # [batch_size, 1, max_text_len]
        text_feats = text_feats.view(-1, text_max_len, text_feat_size)  # [batch_size, text_max_len, text_feat_size]
        text_place_context = torch.bmm(img_place_text_att_dist, text_feats)  # [batch_size, 1, text_feat_size]
        text_place_context = text_place_context.squeeze(1)  # [batch_size, text_feat_size]

        '''----------------------------combined_features----------------------------'''
        combined_features = torch.cat([img_object_context, img_place_context, text_object_context, text_place_context], dim=1)
        combined_features = self.linear(combined_features)
        combined_features = self.dropout(combined_features)
        return combined_features

class MyMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_kv, dropout=0.1, need_mask=False, is_regu=False, interaction_type=None):
        super(MyMultiHeadAttention, self).__init__()
        print('-----------------------------Multi-modal features interaction by Multihead attention: {}-------------------'.format(interaction_type))
        self.need_mask = need_mask
        self.is_regu = is_regu
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_kv, d_kv, dropout=dropout, is_regu=is_regu)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_model, dropout=dropout)

    def forward(self, q, k, v, mask=None):
        # q: [batch_size, d_model] ==>  k: [batch_size, 1, d_model]
        # mask: [batch_size, seq_len] == > [batch_size, 1, seq_len]
        # when there is only one query, we need to expand the dimension
        # print('=================q q q==========================')
        # print(q.size())
        if len(q.shape) == 2:
            q = q.unsqueeze(1)
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len]
        if self.need_mask:
            ##assert作用：检查条件，不符合就终止程序, 即self.mask需要与mask保持一致，即self.need_mask==True, 需要mask is not None, 否则终止程序
            assert mask is not None, 'Please pass the attention mask to the multi-head'
        if self.is_regu:
            enc_output, enc_slf_attn, head_diff = self.slf_attn(q, k, v, mask)
        else:
            enc_output, enc_slf_attn = self.slf_attn(q, k, v, mask)
        enc_output = self.pos_ffn(enc_output)

        # enc_output: [batch_size, 1, d_model] ==>  k: [batch_size, d_model]
        enc_output = enc_output.squeeze(1)
        if self.is_regu:
            return enc_output, enc_slf_attn, head_diff
        return enc_output, enc_slf_attn

class AnotherMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(AnotherMultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        '''
        q: [batch, 1, 512] 
        k, v: [batch, 1, 512]
        '''
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k) ###[bts, 1, 4, 128]
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k) ###[bts, 1, 4, 128]
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v) ###[bts, 1, 4, 128]

        q = q.permute(0, 2, 1, 3).contiguous().view(-1, len_q, d_k)  # [bts, 4, 1, 128]
        k = k.permute(0, 2, 1, 3).contiguous().view(-1, len_k, d_k)  # [bts, 4, 1, 128]
        v = v.permute(0, 2, 1, 3).contiguous().view(-1, len_v, d_v)  # [bts, 4, 1, 128]

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        
        output, attn = self.attention(q, k, v, mask=mask) ###[bts*4,1,128]*[bts*4,128,1]-->[bts*4,1,1]*[bts*4,1,128]-->[bts*4,1,128]
        
        output = output.view(sz_b, n_head, len_q, d_v)  # [bts, 4, 1, 128]

        output = output.permute(0, 2, 1, 3).contiguous()  ###[bts, 1, 4, 128]

        output = output.view(sz_b, len_q, -1)  ###[bts, 1, 4*128]

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn



class MyAnotherMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_kv, dropout=0.1, need_mask=False, interaction_type=None):
        super(MyAnotherMultiHeadAttention, self).__init__()
        print('-----------------------------Multi-modal features interaction by Another Multihead attention: {}-------------------'.format(interaction_type))
        self.need_mask = need_mask
        self.slf_attn = AnotherMultiHeadAttention(n_head, d_model, d_kv, d_kv, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_model, dropout=dropout)
    
    def forward(self, q, k, v, mask=None):
        # q: [batch_size, d_model] ==>  k: [batch_size, 1, d_model]
        # mask: [batch_size, seq_len] == > [batch_size, 1, seq_len]
        # when there is only one query, we need to expand the dimension
        # print('=================q q q==========================')
        # print(q.size())
        if len(q.shape) == 2:
            q = q.unsqueeze(1)
        if len(k.shape) == 2:
            k = k.unsqueeze(1)
        if len(v.shape) == 2:
            v = v.unsqueeze(1)

        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len]
        if self.need_mask:
            ##assert作用：检查条件，不符合就终止程序, 即self.mask需要与mask保持一致，即self.need_mask==True, 需要mask is not None, 否则终止程序
            assert mask is not None, 'Please pass the attention mask to the multi-head'
        
        enc_output, enc_slf_attn = self.slf_attn(q, k, v, mask)
        enc_output = self.pos_ffn(enc_output)

        # enc_output: [batch_size, 1, d_model] ==>  k: [batch_size, d_model]
        enc_output = enc_output.squeeze(1)
        return enc_output, enc_slf_attn

