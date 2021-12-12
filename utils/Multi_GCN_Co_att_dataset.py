import torch.utils.data as data
import json
import os
import subprocess
from PIL import Image
import numpy as np
import torch
import pickle
from utils.util import *
import csv
import torch.nn as nn
import word2vec
import time

class Tumblr_Dataset(data.Dataset):
    def __init__(self, root, dataset, 
                text_min_count, vocab=None,
                transform=None, phase='train', object_inp_name=None, place_inp_name=None):
                
        allowed_data = ['tumblr']
        if dataset not in allowed_data:
            raise ValueError('currently allowed data: %s' % ','.join(allowed_data))
        else:
            self.dataset = dataset
        '''---------------text------------------'''
        self.text_min_count = text_min_count
        self.root = root ##data_root_path: data
        self.phase = phase
        ###glove/glove.6B.200d.txt
        self.word2vec_file = 'data/glove/glove.6B.300d.txt'
        ###data/vocab
        self.vocab_root_path = os.path.join(self.root, 'vocab')
        ###data/all_anno_json
        self.data_root_path = os.path.join(self.root, 'all_anno_json')
        ###data/glove_embedding/glove_embedding_(text_min_cont).pkl
        self.trg_glove_embeding_file = os.path.join(self.root, 'glove_embedding')
        os.makedirs(self.trg_glove_embeding_file, exist_ok=True)
        self.all_text = self.get_content(self.phase)
        self.train_all_text = self.get_content('train')
        
        
        if vocab is None:
            self.vocab = []
            try:
                self.get_vocab()
            except FileNotFoundError:
                if phase == 'train':
                    self.build_vocab()
                else:
                    print('Please build vocab by train dataset!!!')
        else:
            self.vocab = vocab
        '''-----------------判断embedding文件存不存在，不存在就创建---------------'''
        self.glove_embedding_path = os.path.join(self.trg_glove_embeding_file, 'glove_embedding_{}.pkl'.format(str(self.text_min_count)))
        if not os.path.exists(self.glove_embedding_path):
            print('====================Building glove_embedding_text_min_count.pkl------------')
            self.embedding_matrix = self.load_emb_for_vocab(word2vec_file=self.word2vec_file, 
                                vocab=self.vocab, 
                                trg_emb_fn=self.glove_embedding_path)

        self.d = dict(zip(self.vocab, range(len(self.vocab))))
        self.pad_idx = self.d['PAD'] ###PAD在vocab中的id，即index

        '''-----------------------------image----------------------'''
        self.data_list = []
        self.transform = transform
        self.get_anno()

        ###object_inp
        self.object_inp_name = object_inp_name
        with open(object_inp_name, 'rb') as f:
            print('----------------------loading object_inp---------------------')
            self.object_inp = pickle.load(f)
            self.object_inp = torch.from_numpy(np.array(self.object_inp))
        

        ###place_inp
        with open(place_inp_name, 'rb') as f:
            print('----------------------loading place_inp---------------------')
            self.place_inp = pickle.load(f)
            self.place_inp = torch.from_numpy(np.array(self.place_inp))
        self.place_inp_name = place_inp_name

    def get_content(self, phase):
        all_text =[]
        text_data_path = os.path.join(self.data_root_path,'{}_all_anno.json'.format(phase))
        with open(text_data_path, 'r') as f:
            for line in f:
                json_line = json.loads(line)
                text = json_line['text']
                all_text.append(text)
        return all_text

    def word2id(self, word):
        try:
            result = self.d[word]
        except KeyError:
            result = self.d['UNK']
        return result
    def get_vocab(self):
        vocab_path = os.path.join(self.vocab_root_path, 'vocab-' + str(self.text_min_count)+'.txt')
        with open(vocab_path, 'r') as f:
            print('------------------------vocab path is: {}-------------------'.format(vocab_path))
            print('------------------------geting vocab-----------------------')
            vocab = f.read()
            self.vocab = vocab.split('\n')
            print('the length of vocab is: ', len(self.vocab))

    def build_vocab(self):
        print('------------------building vocab,使用train数据---------------------')
        vocab = []
        for text in self.train_all_text:
            words = text.split(' ')
            for word in words:
                if word not in vocab:
                    vocab.append(word)

        freq = dict(zip(vocab, [0 for i in range(len(vocab))]))
        for text in self.train_all_text:
            words = text.split(' ')
            for word in words:
                freq[word] += 1
        with open(os.path.join(self.vocab_root_path, 'freq.csv'), 'w') as f:
            print('-------------save csv----------')
            writer = csv.writer(f)
            results = list(zip(freq.keys(), freq.values()))
            writer.writerows(results)

        results = []
        for word in freq.keys():
            if freq[word] < self.text_min_count:
                continue
            else:
                results.append(word)

        results.insert(0, 'UNK')
        with open(os.path.join(self.vocab_root_path, 'vocab-' + str(self.text_min_count)+'.txt'), 'w') as f:
            f.write('\n'.join(results))

        self.vocab = results
        
    
    def load_emb_for_vocab(self, word2vec_file, vocab, trg_emb_fn, emb_size=300):
        '''
        将vocab对应的embedding存储, 因为只需要在建立vocab的时候为vocab加载embedding，故只需在build_vocab()中
        使用，在此处直接传入vocab列表即可
        word2vec_file: 存储预训练的glove_embedding文件的路径，glove/glove.6B.300d.txt, 使用word2vec模型加载
        glove与word2vec区别是，word2vec 需要在第一行插入单词数及维度，如：400000 300
        vocab_pt_fn: 存储vocab文件的路径，data/dataset/vocab/vocab_(text_min_count).txt
        vocab: vocab列表，
        trg_emb_fn: 存储对应vocab的embedding文件的路径 data/dataset/glove_embedding/glove_embedding_(text_min_count).pkl
        emb_size=200
        '''
        vocab_size = len(vocab)
        t0 = time.time()

        model = word2vec.load(word2vec_file)
        embedding_matrix = []
        for word in vocab:
            try:
                embedding_matrix.append(model[word])
            except KeyError:
                print('word: %s is not fund' % (word))
                embedding_matrix.append(model['the'])
        embedding_matrix = np.array(embedding_matrix)
        print('Find %d from glove embedding, takes %.2f seconds' % (vocab_size, time.time() - t0))
        print('the shape of embedding_matrix is: ',np.shape(embedding_matrix))

        with open(trg_emb_fn, 'wb') as f:
            pickle.dump(embedding_matrix, f)
        print('Saving embedding into %s' % trg_emb_fn)

        return embedding_matrix
        

    def get_anno(self):
        list_path = os.path.join(self.data_root_path, '{}_all_anno.json'.format(self.phase))
        print('----------------------{}_json_path---------------'.format(self.phase))
        print(list_path)
        json_lines = []
        text_list = []
        text2id_list = []
        with open(list_path, 'r') as f:
            for line in f:
                json_line = json.loads(line)
                # print(json_line)
                text_list.append(json_line['text'])
                json_lines.append(json_line)
        
        self.data_list = json_lines
        ##转为单词index序列
        for text in text_list:
            text2id = list(map(lambda x: self.word2id(x), text.split(' ')))
            text2id_list.append(text2id)
        input_list_lens = [len(l) for l in text2id_list]

        self.text2id_list = text2id_list ##所有文本序列，word2id的列表
        self.input_list_lens = input_list_lens ##所有文本序列长度列表
        self.text_max_length = max(input_list_lens) ##最长文本序列长度

        self.cat2idx = json.load(open(os.path.join('data/label.json'), 'r'))
        self.num_classes = len(self.cat2idx)
        

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        item = self.data_list[index]
        return self.get(item)

    def _pad(self, input_list):
        ##处理整个文本序列列表，即整个数据集
        # 将文本padding至最长文本序列长度，input_list: 所有文本序列，word2id的列表
        input_list_lens = [len(l) for l in input_list]
        max_seq_len = max(input_list_lens)
        padded_batch = self.pad_idx * np.ones((len(input_list), max_seq_len))

        for j in range(len(input_list)):
            current_len = input_list_lens[j]
            padded_batch[j][:current_len] = input_list[j]

        padded_batch = torch.LongTensor(padded_batch)
        '''
        torch.ne(input, other, out=Tensor)-->Tensor, ne: not equal to; other可以是一个数或和input相同形状和类型的张量
        逐元素比较input和other, 返回是torch.BoolTensor,包含了每个位置的比较结果(如果tensor!=other为True，返回1)
        '''
        input_mask = torch.ne(padded_batch, self.pad_idx)
        input_mask = input_mask.type(torch.FloatTensor)

        return padded_batch, input_list_lens, input_mask
    
    def _padding(self, content):
        '''
        每次仅处理一条文本，适用于我自己代码，根据self._pad()改编
        content_padding:动态计算文本序列长度，以最大文本序列长度填充后的序列。并保存mask
        '''
        current_len = len(content)
        content_padding= self.pad_idx * np.ones((1, self.text_max_length))###[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]
        content_padding = content_padding.reshape(-1)
        content_padding[:current_len] = content
        content_padding = torch.LongTensor(content_padding)
        text_mask = torch.ne(content_padding, self.pad_idx)
        text_mask = text_mask.type(torch.FloatTensor)
        return content_padding, current_len, text_mask

    def get(self, item):
        self.id = item['id']
        image_path = item['image']
        self.text = item['text'] ##为文本内容
        ##转为单词index序列
        self.content = list(map(lambda x: self.word2id(x), self.text.split(' ')))
        content_padding, current_len, text_mask = self._padding(self.content)
        self.content = content_padding
        self.content = np.array(self.content).T
        self.label = self.cat2idx[item['label']]
        self.objects = sorted(item['objects'])
        self.places = sorted(item['places'])
        img = Image.open(image_path).convert('RGB')
        
        # print('--------------------type(img)-----------------------------')
        # print(type(img))
        if self.transform is not None:
            img = self.transform(img)
        return (self.id, self.text, self.content, current_len, text_mask, img, image_path, self.object_inp, self.place_inp), self.label

if __name__ == '__main__':
    path = 'data'
    dataset = 'tumblr'
    text_min_count = 5
    

    test_dataset = Tumblr_Dataset(path, dataset, 
                                text_min_count, vocab=None,
                                phase='test', object_inp_name='data/glove/object_glove_word2vec.pkl', 
                                                    place_inp_name='data/glove/place_glove_word2vec.pkl')
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size= 64, shuffle=False)
    # for i, (input_, target) in enumerate(test_loader):
    #     print(input_[0])
    #     print(input_[1])
    #     print(input_[2])
    #     print(input_[3])
    #     print(input_[4])
    #     print(input_[5])

