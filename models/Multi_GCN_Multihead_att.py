import torchvision.models as models
from torch.nn import Parameter
from models.Text_GCN import Model as Text_GCN_Model
from utils.util import *
from utils.vocab import get_vocab_list
from utils.pmi import cal_PMI
import torch
import torch.nn as nn
import math
import numpy
import dgl
import word2vec
import torch.nn.functional as F
import numpy as np
from models.moudles import CoAttention, MyMultiHeadAttention, MyAnotherMultiHeadAttention



###使用标签矩阵作为query
save_label_pkl_path = 'data/glove/tumblr_label_glove.pkl'
def get_glove_embedding(glove_file):
    ###读取pkl文件
    f=open(glove_file,'rb')
    glove_embedding=pickle.load(f)    #读出文件的数据个
    return glove_embedding
glove_label_embedding = get_glove_embedding(save_label_pkl_path)
glove_label_embedding = torch.from_numpy(numpy.array(glove_label_embedding))


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class Attention(nn.Module):
    '''
    用于计算情感感知的图像GCN表示的注意力
    '''
    def __init__(self, hid_dim, image_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim ###300, 标签的维度，query的维度
        self.n_heads = n_heads ###
        
        # d_model // h 仍然是要能整除，换个名字仍然意义不变
        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim) ###label序列，300dim-->300, [3,300]
        self.w_k = nn.Linear(image_dim, hid_dim) ###图像数据，365 -->300, [batch_size, 365]-->[batch_size,300]
        self.w_v = nn.Linear(image_dim, hid_dim) ###图像数据，365 -->300, [batch_size, 365]-->[batch_size,300]

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)
        device = torch.device('cuda:0' )
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        '''
        query: label [7,300]
        key: image [batch_size, 2048]
        value: image [batch_size, 2048]
        '''
       # Q,K,V计算与变形：
        bsz = key.shape[0]###batch_size

        Q = self.w_q(query.float())###query为label序列，300dim-->300, [3,300]
        K = self.w_k(key)###图像数据，num_classes-->300, [batch_size, 80 or 365]-->[batch_size,300]
        V = self.w_v(value)###图像数据，num_classes-->300, [batch_size, 80 or365]-->[batch_size,300]

        Q = Q.view(7, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 1, 2)###[7, 5, 60] 
        K = K.view(bsz, self.n_heads, self.hid_dim // 
                   self.n_heads).permute(0, 1, 2)###[batch_size, 5, 60]
        V = V.view(bsz, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 1, 2)###[batch_size, 5, 60]
        # Q, K相乘除以scale，这是计算scaled dot product attention的第一步

        K = torch.unsqueeze(K, dim=1) ##[batch_size, 1, 5, 60] heads=5
        K_cat = torch.cat((K, K, K, K, K, K, K), dim=1)###[batch_size, 7, 5, 60]
        energy_0 = torch.mul(Q, K_cat[0]) ###[7, 5, 60]
        energy_0 = torch.unsqueeze(energy_0, dim=0) ###[1, 7, 5, 60]
        energy_all = energy_0
        for i in range(1, bsz):
            energy_all= torch.cat((energy_all, torch.unsqueeze(torch.mul(Q, K_cat[i]), dim=0)), dim=0)
            ##energy_all.torch size: [batch_size, 7, 5, 60]
        energy = energy_all/self.scale
        
        # 如果没有mask，就生成一个
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # 然后对Q,K相乘的结果计算softmax加上dropout，这是计算scaled dot product attention的第二步：
        attention = self.do(torch.softmax(energy, dim=-1))

        # 第三步，attention结果与V相乘
        V = torch.unsqueeze(V, dim=1) ##[batch_size, 1, 5, 60] heads=5
        V_cat = torch.cat((V, V, V, V, V, V, V), dim=1)###[batch_size, 7, 5, 60]
        x = torch.mul(attention, V_cat) 
        x = x.contiguous()
        x = x.view(bsz, 7, self.n_heads * (self.hid_dim // self.n_heads)) ###[batch_size, 7, 300]
        x = self.fc(x) ###[batch_size, 7, 300]
        return x ###300

class Multi_GCN_Multihead_Att(nn.Module):
    def __init__(self, opt,
                 num_labels, 
                 text_model,object_model, place_model, 
                 object_num_classes, place_num_classes, 
                 object_t=0, place_t=0,
                 in_channel=300,
                 object_adj_file=None, place_adj_file=None):
        super(Multi_GCN_Multihead_Att, self).__init__()
        print('-----------opt-------------')
        print(opt)
        self.emb_path = opt['emb_path'] ###glove_embedding
        self.bidirectional = opt['bidirectional']
        self.num_directions = 2 if self.bidirectional else 1
        self.hidden_size = opt['hidden_size']
        self.bi_hidden_size = self.num_directions * opt['hidden_size']
        opt['bi_hidden_size'] = self.bi_hidden_size
        '''
        在本代码中，self.d_model ==self.bi_hidden_size, 也可不一致，那就需要修改，models.multi_hean_att.submodules.MultiHeadAttention中的代码，
        修改输入的q和k,v的第三维的特征维度，其中k,v第三维特征维度为bi_hidden_size, 在此设置的bi_hidden_size=300; 
        q第三维特征维度与使用的glove嵌入有关，
        '''
        self.d_model = self.bi_hidden_size
        self.pad_idx = 0
        self.stack_num = opt['stack_num'] ###multihead_att堆叠的层数 
        self.n_head = opt['n_head'] ###Multihead_att多头注意力头数
        self.d_kv = opt['d_kv'] ##多头注意力，每一头的维度
        self.is_regu = opt['is_regu']


        self.embedding = nn.Embedding(
            opt['vocab_size'],
            opt['emb_size'],
            padding_idx=self.pad_idx ###vocab['PAD']=0
        )
        self.init_weights(opt['emb_type'], self.pad_idx)

        self.rnn = nn.GRU(input_size=opt['emb_size'], 
                          hidden_size=opt['hidden_size'], 
                          num_layers=opt['num_layers'],
                          bidirectional=opt['bidirectional'], 
                          batch_first=True, 
                          dropout=opt['dropout'])

        self.lstm = nn.LSTM(input_size=opt['emb_size'], 
                          hidden_size=opt['hidden_size'], 
                          num_layers=opt['num_layers'],
                          bidirectional=opt['bidirectional'], 
                          batch_first=True, 
                          dropout=opt['dropout'])
        
        self.object_gate = nn.Linear(self.bi_hidden_size*2, self.bi_hidden_size) ###600--->300
        self.place_gate = nn.Linear(self.bi_hidden_size*2, self.bi_hidden_size) ###600--->300

        '''------------------------------Multi-modal features interaction by Multi_head Attention-----------------------------'''
        '''
        +++++++++++++++++++++++img_object_text_multi_head_att+++++++++++++++++++++
        query：image_object,为2维特征, 即object_x_attention, [batch_size, 300], 
        key：text_rnn_feat(text_memory_bank),为3维特征，[batch_size, text_max_len, bi_hidden_size]
        value: text_rnn_feat(text_memory_bank),为3维特征，[batch_size, text_max_len, bi_hidden_size]
        output: [batch_size, 300], 与query一致
        '''
        self.img_object_text_multi_head_att = nn.ModuleList(
                    [MyMultiHeadAttention(self.n_head, self.d_model, self.d_kv, dropout=opt['dropout'], need_mask=True,
                                          is_regu=self.is_regu, interaction_type='img_object_text')
                     for _ in range(self.stack_num)])
        

        '''
        +++++++++++++++++++++++text_object_text_multi_head_att+++++++++++++++++++++
        query：self.img_object_text_multi_head_att,为2维特征, 即img_object_text_enc_output, [batch_size, 300], 
        key：text_feature,为2维特征，[batch_size, 1, bi_hidden_size]
        value: text_feature,为2维特征，，[batch_size, 1, bi_hidden_size]
        output: [batch_size, 300], 与query一致
        '''
        self.text_object_text_multi_head_att = MyAnotherMultiHeadAttention(self.n_head, 
                                                                           self.d_model, 
                                                                           self.d_kv, 
                                                                           dropout=opt['dropout'], 
                                                                           need_mask=False,
                                                                           interaction_type='text_object_text')

        '''
        +++++++++++++++++++++++img_place_text_multi_head_att+++++++++++++++++++++
        query：image_place,为2维特征, 即place_x_attention, [batch_size, 300]
        key：text_rnn_feat(text_memory_bank),为3维特征，[batch_size, text_max_len, bi_hidden_size]
        value: text_rnn_feat(text_memory_bank),为3维特征，[batch_size, text_max_len, bi_hidden_size]
        output: [batch_size, 300], 与query一致
        '''
        self.img_place_text_multi_head_att = nn.ModuleList(
                    [MyMultiHeadAttention(self.n_head, self.d_model, self.d_kv, dropout=opt['dropout'], need_mask=True,
                                          is_regu=self.is_regu, interaction_type='img_place_text')
                     for _ in range(self.stack_num)])

        '''
        +++++++++++++++++++++++text_place_text_multi_head_att+++++++++++++++++++++
        query：self.img_place_text_multi_head_att,为2维特征, 即img_place_text_enc_output, [batch_size, 300], 
        key：text_rnn_feat(text_memory_bank),为3维特征，[batch_size, text_max_len, bi_hidden_size]
        value: text_rnn_feat(text_memory_bank),为3维特征，[batch_size, text_max_len, bi_hidden_size]
        output: [batch_size, 300], 与query一致
        '''
        self.text_place_text_multi_head_att = MyAnotherMultiHeadAttention(self.n_head, 
                                                                           self.d_model, 
                                                                           self.d_kv, 
                                                                           dropout=opt['dropout'], 
                                                                           need_mask=False,
                                                                           interaction_type='text_place_text')
        
        '''
        +++++++++++++++++++++++text_img_object_multi_head_att+++++++++++++++++++++
        query：text_feature,为2维特征, [batch_size, 300]
        key：img_object_memory_bank,为3维特征，[batch_size, 196, bi_hidden_size]
        value: img_object_memory_bank,为3维特征，[batch_size, 196, bi_hidden_size]
        output: [batch_size, 300], 与query一致
        '''
        self.text_img_object_multi_head_att = nn.ModuleList(
                    [MyMultiHeadAttention(self.n_head, self.d_model, self.d_kv, dropout=opt['dropout'], need_mask=False,
                    interaction_type='text_img_object')
                     for _ in range(self.stack_num)])
        
        '''
        +++++++++++++++++++++++text_img_place_multi_head_att+++++++++++++++++++++
        query：text_feature,为2维特征, [batch_size, 300]
        key：img_place_memory_bank,为3维特征，[batch_size, 196, bi_hidden_size]
        value: img_place_memory_bank,为3维特征，[batch_size, 196, bi_hidden_size]
        output: [batch_size, 300], 与query一致
        '''
        self.text_img_place_multi_head_att = nn.ModuleList(
                    [MyMultiHeadAttention(self.n_head, self.d_model, self.d_kv, dropout=opt['dropout'], need_mask=False,
                    interaction_type='text_img_place')
                     for _ in range(self.stack_num)])

        self.liner_img_object = nn.Linear(2048, self.bi_hidden_size)
        self.liner_img_place = nn.Linear(2048, self.bi_hidden_size)

        ###text features
        self.text_features = text_model
        
        ###object features
        self.object_features = nn.Sequential(
            object_model.conv1,
            object_model.bn1,
            object_model.relu,
            object_model.maxpool,
            object_model.layer1,
            object_model.layer2,
            object_model.layer3,
            object_model.layer4,
        )
        ###place features
        self.place_features = nn.Sequential(
            place_model.conv1,
            place_model.bn1,
            place_model.relu,
            place_model.maxpool,
            place_model.layer1,
            place_model.layer2,
            place_model.layer3,
            place_model.layer4,
        )
        
        self.num_labels = num_labels
        self.object_num_classes = object_num_classes ###object 类别数目:80
        self.place_num_classes = place_num_classes ###place 类别数目:365
        self.object_t = object_t
        self.place_t = place_t

        self.pooling = nn.MaxPool2d(14, 14)
        ###图像GCN操作
        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.leakyrelu = nn.LeakyReLU(0.2)
        ##激活函数
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        ##图像注意力
        self.object_attention = Attention(hid_dim=300, image_dim=self.object_num_classes, n_heads=5, dropout=0.5)
        self.place_attention = Attention(hid_dim=300, image_dim=self.place_num_classes, n_heads=5, dropout=0.5)
      
         ###对object图像特征进行线性变换
        self.object_linear_1 = nn.Linear(2048, 1024) 
        self.object_linear_2 = nn.Linear(1024, 512)
        self.object_linear_3 = nn.Linear(512, 256)
        
        self.object_linear_5 = nn.Linear(300, 100)###注意力之后的纬度变换
        self.object_x_linear = nn.Linear(700, 300)

        ###对place图像特征进行线性变换
        self.place_linear_1 = nn.Linear(2048, 1024) 
        self.place_linear_2 = nn.Linear(1024, 512)
        self.place_linear_3 = nn.Linear(512, 256)

        self.place_linear_5 = nn.Linear(300, 100)###注意力之后的维度变换
        self.place_x_linear = nn.Linear(700, 300)
    
        self.dropout = nn.Dropout(0.5) ###不保留节点数的比例

        ###multi linear
        self.multi_linear_1 = nn.Linear(1200, self.bi_hidden_size)
        self.multi_linear_2 = nn.Linear(self.bi_hidden_size, num_labels)

        ###object_Adj
        object_adj, object_nums = gen_A(object_num_classes, self.object_t, object_adj_file)
        print('---------------  object_adj----------------------')
        print(object_adj.shape)
        self.object_A = Parameter(torch.from_numpy(object_adj).float())

        ###place_Adj
        place_adj, place_nums = gen_A(place_num_classes, self.place_t, place_adj_file)
        print('---------------place_adj----------------------')
        print(place_adj.shape)
        self.place_A = Parameter(torch.from_numpy(place_adj).float())

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def init_weights(self, emb_type, pad_idx):
        """Initialize weights."""
        if emb_type == 'random':
            initrange = 0.1
            self.embedding.weight.data.uniform_(-initrange, initrange)
        else:
            with open(self.emb_path, 'rb') as f:
                weights = pickle.load(f)
            self.embedding.weight.data = torch.Tensor(weights)
            # self.embedding.weight.requires_grad = False
            print('Load glove embedding!')
        self.embedding.weight.data[pad_idx] = 0
    
    def get_text_memory_bank(self, text, text_lens, return_last_state=True):
        # embed the text post with embedding and Bi-GRU layers
        batch_size, max_text_len = list(text.size())
        # print('----------------------max_text_len---------------')
        # print(max_text_len)
        text_embed = self.embedding(text)  # [batch, text_len, emb_size]
        '''
        nn.utils.rnn.pack_padded_sequence:压缩padding部分，pack之后，原来填充的 PAD（一般初始化为0）占位符被删掉了。
        输入的形状可以是(T×B×* )。T是最长序列长度，B是batch size，*代表任意维度(可以是0)。如果batch_first=True的话，那么相应的 input size 就是 (B×T×*)。
        '''
        packed_input_text = nn.utils.rnn.pack_padded_sequence(text_embed, text_lens, batch_first=True, enforce_sorted=False)
        # memory_bank, enc_final_state = self.rnn(packed_input_text)
        memory_bank, (enc_final_state, c_n) = self.lstm(packed_input_text)
        # ([batch, seq_len, num_directions*hidden_size], [num_layer * num_directions, batch, hidden_size])
        '''
        nn.utils.rnn.pad_packed_sequence：这个操作和pack_padded_sequence()是相反的。把压紧的序列再填充回来。填充时会初始化为0。
        返回的Varaible的值的size是 T×B×*, T 是最长序列的长度，B 是 batch_size,如果 batch_first=True,那么返回值是B×T×*。
        '''
        memory_bank, _ = nn.utils.rnn.pad_packed_sequence(memory_bank, batch_first=True, total_length=max_text_len)  # unpack (back to padded)
        memory_bank = memory_bank.contiguous()
        # print('----------------------text_memory_bank.size()---------------')
        # print(memory_bank.size())
        assert memory_bank.size() == torch.Size([batch_size, max_text_len, self.bi_hidden_size])

        if self.bidirectional:
            # [batch, hidden_size*2]
            enc_last_layer_final_state = torch.cat((enc_final_state[-1, :, :], enc_final_state[-2, :, :]), 1)
        else:
            # [batch, hidden_size]
            enc_last_layer_final_state = enc_final_state[-1, :, :]  
        if return_last_state:
            return memory_bank, enc_last_layer_final_state
        return memory_bank

    def get_img_object_memory_bank(self, img_object_feats):
        '''
        read image_object visual feature and map them to bi_hidden_size
        img_object_feats: Resnet101:[batch_size, 2048, 14, 14]
        '''
        batch_size = img_object_feats.shape[0]
        img_object_feats = img_object_feats.view([img_object_feats.size(0), 2048, -1]).permute([0, 2, 1]) # [batch_size, 196, 2048]
        # print('---------------img_object_feats.size()---------------')
        # print(img_object_feats.size())
        img_object_feats = img_object_feats.reshape(-1, img_object_feats.shape[2])##[batch_size*196, 2048]
        # print('---------------img_object_feats.size()---------------')
        # print(img_object_feats.size())
        img_object_feats = self.liner_img_object(img_object_feats)##[batch_size*196, 2048]--->[batch_size*196, bi_hidden_size]
        img_object_feats = img_object_feats.view(batch_size, -1, img_object_feats.shape[-1])  # [batch_size, 196, bi_hidden_size]
        # print('---------------img_object_feats.size()---------------')
        # print(img_object_feats.size())
        return img_object_feats # [batch_size, 196, bi_hidden_size]

    def get_img_place_memory_bank(self, img_place_feats):
        '''
        read image_place visual feature and map them to bi_hidden_size
        img_place_feats: [batch_size, 2048, 14, 14]
        '''
        batch_size = img_place_feats.shape[0]
        img_place_feats = img_place_feats.view([img_place_feats.size(0), 2048, -1]).permute([0, 2, 1]) # [batch_size, 196, 2048]
        img_place_feats = img_place_feats.reshape(-1, img_place_feats.shape[2])##[batch_size*196, 2048]
        img_place_feats = self.liner_img_place(img_place_feats)##[batch_size*256, 2048]--->[batch_size*196, bi_hidden_size]
        img_place_feats = img_place_feats.view(batch_size, -1, img_place_feats.shape[-1])  # [batch_size, 196, bi_hidden_size]
        return img_place_feats  # [batch_size, 196, bi_hidden_size]


    def forward(self, text, text_lens, text_mask, object_feature, place_feature, object_inp, place_inp, return_last_state=True):
        '''
        text:已经被word2id表示的文本序列
        text_lens: 存储未padding文本长度的列表
        text_mask: bool_tensor, 表示padding与否
        objec_feature:  torch.Size([3, 448, 448]) '448'为image_size
        place_feature: torch.Size([3, 448, 448]) '448'为image_size 
        object_inp:input:torch.Size([80, 300])  
        place_inp:input:torch.Size([365, 300])  
        虽然object_feature与place_feature维度一致，但是值不一样
        '''
        '''--------------------text_feature--------------------'''
        '''+++++++++++++通过TextLevelGCN得到++++++++++++++++++'''
        text_feature = self.text_features(text) ###[batch, 300]
        '''----------------------获取三维文本表示，用于之后的多模态交互-------'''
        ###([batch, seq_len, num_directions*hidden_size], [batch, hidden_size*num_directions])
        text_memory_bank, text_encoder_final_state = self.get_text_memory_bank(text, text_lens, return_last_state)
        
        '''-----------------object_feature--------------------'''
        self.object_feature = self.object_features(object_feature)###torch.Size([8, 2048, 14, 14]),8:batch_size
        '''-------------------获取imag_object_3dim特征，用于之后的多模态交互--------'''
        img_object_memory_bank = self.get_img_object_memory_bank(self.object_feature)###[bts, 14*14, bi_hidden_size]

        object_feature = self.pooling(self.object_feature)###torch.Size([8, 2048, 1, 1])
        object_feature = object_feature.view(object_feature.size(0), -1)######torch.Size([8, 2048])
        ###x.size(0)指batchsize的值；
        ####x = x.view(x.size(0), -1)  这句话的出现就是为了将前面多维度的tensor展平成一维

        ###object_inp
        object_inp = object_inp[0] ###80
        object_adj = gen_adj(self.object_A).detach()
        query = glove_label_embedding

        ###为了获得object classes 之间的依赖关系，进行GCN操作
        device = torch.device('cuda:0' )
        object_inp =  object_inp.to(device) ###[80,300]
        object_adj =  object_adj.to(device)###[80,80]
        query = query.to(device)

        object_x = self.gc1(object_inp, object_adj)
        object_x = self.leakyrelu(object_x)
        object_x = self.gc2(object_x, object_adj)
        object_x = object_x.transpose(0, 1)###torch.Size([2048, 80])
        object_x = torch.matmul(object_feature, object_x) ###torch.Size([batch_size, 80])完成图像视觉特征与GCN object 多标签的融合，矩阵乘法,8为batch_size
        '''-----------object_attention----------------------'''
        object_x_attention = self.object_attention(query=query, key=object_x, value=object_x) ###[batch_size, 3, 300]
        object_x_attention = self.object_linear_5(object_x_attention)###[batch_size, 7, 100]
        object_x_attention = object_x_attention.view(object_feature.size(0), -1)###[batch_size, 700]
        object_x_attention = self.object_x_linear(object_x_attention)

        '''-----------------place_feature--------------------'''
        self.place_feature = self.place_features(place_feature)###torch.Size([8, 2048, 14, 14]),8:batch_size
        '''-------------------获取imag_place_3dim特征，用于之后的多模态交互--------'''
        img_place_memory_bank = self.get_img_place_memory_bank(self.place_feature)###[bts, 14*14, bi_hidden_size]
        
        place_feature = self.pooling(self.place_feature)###torch.Size([8, 2048, 1, 1])
        place_feature = place_feature.view(place_feature.size(0), -1)######torch.Size([8, 2048])
        ###place_inp
        place_inp = place_inp[0] ###365
        place_adj = gen_adj(self.place_A).detach()

        ###为了获得object classes 之间的依赖关系，进行GCN操作
        device = torch.device('cuda:0' )
        place_inp =  place_inp.to(device) ###[365,300]
        place_adj =  place_adj.to(device)###[365,365]
        place_x = self.gc1(place_inp, place_adj)
        place_x = self.leakyrelu(place_x)
        place_x = self.gc2(place_x, place_adj)
        place_x = place_x.transpose(0, 1)###torch.Size([2048, 365])
        place_x = torch.matmul(place_feature, place_x) ###torch.Size([8, 365])完成图像视觉特征与GCN place 多标签的融合，矩阵乘法,8为batch_size
       
        '''-----------place_attention----------------------'''
        place_x_attention = self.place_attention(query=query, key=place_x, value=place_x) ###[batch_size, 3, 300]
        place_x_attention = self.place_linear_5(place_x_attention)###[batch_size, 7, 100]
        place_x_attention = place_x_attention.view(place_feature.size(0), -1)###[batch_size, 700]
        place_x_attention = self.place_x_linear(place_x_attention)
        
        '''--------------------------multi feature: img_object_text_multi_head_att  -------------------------------'''
        img_object_text_enc_output = object_x_attention
        for img_object_text_enc_layer in self.img_object_text_multi_head_att:
            img_object_text_enc_output, _ = img_object_text_enc_layer(q=img_object_text_enc_output,
                                                                      k=text_memory_bank,
                                                                      v=text_memory_bank,
                                                                      mask=text_mask) ###[batch_size, 300]

        '''--------------------------multi feature: text_object_text_multi_head_att  -------------------------------'''
        # text_object_text_enc_output,_ = self.text_object_text_multi_head_att(q=text_feature,
        #                                                                     k=img_object_text_enc_output,
        #                                                                     v=img_object_text_enc_output)

        '''--------------------------multi feature: img_place_text_multi_head_att  -------------------------------'''
        img_place_text_enc_output = place_x_attention
        for img_place_text_enc_layer in self.img_place_text_multi_head_att:
            img_place_text_enc_output, _ = img_place_text_enc_layer(q=img_place_text_enc_output,
                                                                    k=text_memory_bank,
                                                                    v=text_memory_bank,
                                                                    mask=text_mask) ###[batch_size, 300]

        '''--------------------------multi feature: text_place_text_multi_head_att  -------------------------------'''
        # text_place_text_enc_output,_ = self.text_place_text_multi_head_att(q=text_feature,
        #                                                                     k=img_place_text_enc_output,
        #                                                                     v=img_place_text_enc_output)

        '''--------------------------multi feature: text_img_object_multi_head_att  -------------------------------'''
        text_img_object_enc_output = text_feature
        for text_img_object_enc_layer in self.text_img_object_multi_head_att:
            text_img_object_enc_output, _ = text_img_object_enc_layer(q=text_img_object_enc_output,
                                                                      k=img_object_memory_bank,
                                                                      v=img_object_memory_bank) ###[batch_size, 300]

        '''--------------------------multi feature: text_img_place_multi_head_att  -------------------------------'''
        text_img_place_enc_output = text_feature
        for text_img_place_enc_layer in self.text_img_place_multi_head_att:
            text_img_place_enc_output, _ = text_img_place_enc_layer(q=text_img_place_enc_output,
                                                                    k=img_place_memory_bank,
                                                                    v=img_place_memory_bank) ###[batch_size, 300]
        
        '''---------------------visual object gate------------------------------'''
        # text_object_merge_representation = torch.cat([text_object_text_enc_output, text_img_object_enc_output], dim=1)
        # text_object_gate_value = torch.sigmoid(self.object_gate(text_object_merge_representation ))  # [batch_size, bi_hidden_dim]
        # gated_converted_att_object_embed = torch.mul(text_object_gate_value, text_img_object_enc_output)

        # '''---------------------visual place gate------------------------------'''
        # text_place_merge_representation = torch.cat([text_place_text_enc_output, text_img_place_enc_output], dim=1)
        # text_place_gate_value = torch.sigmoid(self.place_gate(text_place_merge_representation ))  # [batch_size, bi_hidden_dim]
        # gated_converted_att_place_embed = torch.mul(text_place_gate_value, text_img_place_enc_output)



        multi_feature = torch.cat([text_img_object_enc_output, text_img_place_enc_output,
                                   img_object_text_enc_output, img_place_text_enc_output], dim=1) ###[batch_size, 1200]

        multi_feature = self.multi_linear_1(multi_feature) ###1800-->300
        multi_feature = self.dropout(multi_feature)
        # multi_feature = torch.cat([multi_feature, text_feature], dim=1)
        multi_feature = self.multi_linear_2(multi_feature) ###300--->3 (for MVSA, it is 3.)
        return multi_feature

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.text_features.parameters(), 'lr': lr * 10},
                {'params': self.object_features.parameters(), 'lr': lr * lrp},
                {'params': self.place_features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                {'params': self.object_attention.parameters(), 'lr': lr},
                {'params': self.place_attention.parameters(), 'lr': lr},
                {'params': self.lstm.parameters(), 'lr': lr * 10},
                {'params': self.img_object_text_multi_head_att.parameters(), 'lr': lr},
                {'params': self.img_place_text_multi_head_att.parameters(), 'lr': lr},
                {'params': self.text_img_object_multi_head_att.parameters(), 'lr': lr},
                {'params': self.text_img_place_multi_head_att.parameters(), 'lr': lr},
                # {'params': self.text_object_text_multi_head_att.parameters(), 'lr': lr},
                # {'params': self.text_place_text_multi_head_att.parameters(), 'lr': lr},
                ]
def place_resnet(arch = 'resnet50'):
     ###arch: PyTorch Places365 models: AlexNet, ResNet18, ResNet50, DenseNet161.
    # load the pre-trained weights
    model_file = 'weights/%s_places365.pth.tar' % arch
    model = models.__dict__[arch](num_classes=365)
    # print(model)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    return model


def Text_model(data_root_path, vocab_root_path, text_min_count, window_size,
               num_labels, ngram, text_dropout, min_cooccurence):
    ###加载vocab 
    vocab = get_vocab_list(data_root_path, vocab_root_path, text_min_count)
    ##MVSA_simple_windowsize=5,6 MVSA_multiple_windowsize=10,11 
    edges_weights, edges_mappings, count = cal_PMI(data_root_path, 
                                                   vocab_root_path, 
                                                   min_count = text_min_count, 
                                                   phase='train', 
                                                   window_size=window_size,
                                                   min_cooccurence=min_cooccurence)
    # print('---------------------count.type----------------------------')
    # print(type(count))
    text_model = Text_GCN_Model(num_labels, hidden_size_node=300,
               vocab=vocab, n_gram=ngram, drop_out=text_dropout, 
               edges_matrix=edges_mappings, edges_num=count,
               pmi=edges_weights, cuda=True, trainable_edges=True)
    return text_model



def multi_gcn_multihead_att_model(opt,
                           num_labels, 
                           object_num_classes, place_num_classes,object_t, place_t,
                           data_root_path, vocab_root_path, 
                           text_min_count, window_size,
                           ngram, min_cooccurence,
                           text_dropout=0.5,
                           pretrained=True,
                           object_adj_file=None, place_adj_file=None,in_channel=300):
              
    object_model = models.resnet101(pretrained=pretrained)
    place_model = place_resnet()
    text_model = Text_model(data_root_path, vocab_root_path, text_min_count, window_size,
                            num_labels, ngram, text_dropout, min_cooccurence)
    Multi_GCN_Multihead_Att_model = Multi_GCN_Multihead_Att(opt,
                                       num_labels, text_model=text_model,
                                       object_model=object_model, place_model=place_model,
                                       object_num_classes=object_num_classes, place_num_classes=place_num_classes,
                                       in_channel=in_channel,
                                       object_t=object_t, place_t=place_t,
                                       object_adj_file=object_adj_file,place_adj_file=place_adj_file
                                )

    return Multi_GCN_Multihead_Att_model
