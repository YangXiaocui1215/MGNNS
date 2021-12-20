
import numpy as np
import torch
import os
import json
from utils.vocab_new import get_vocab_list

def text_padding(content):
    print('-------------text padding-----------------')
    new_content = []
    for i in range(len(content)):
        sentence = content[i].split(' ')
        if len(sentence)<100:
            sentence = sentence + ['PAD']*(100 - len(sentence))
            new_content.append(sentence)
    return new_content
    
def get_content(data_root_path):
    all_text =[]
    text_data_path = os.path.join(data_root_path, 'all_anno_json', 'train_all_anno.json')
    with open(text_data_path, 'r') as f:
        for line in f:
            json_line = json.loads(line)
            text = json_line['text']
            all_text.append(text)
    return all_text

def cal_PMI(data_root_path, vocab_root_path, min_count, phase='train', window_size=6, min_cooccurence=2):

    vocab=get_vocab_list(data_root_path, vocab_root_path, min_count)
    all_text = get_content(data_root_path)
    ###text padding
    all_text = text_padding(all_text)

    d = dict(zip(vocab, range(len(vocab))))

    pair_count_matrix = np.zeros((len(vocab), len(vocab)), dtype=int)
    word_count =np.zeros(len(vocab), dtype=int)
    print('the shape of word_count: ', np.shape(word_count)) 
    for sentence in all_text:
        # sentence = sentence.split(' ')
        for i, word in enumerate(sentence):
            if word!='PAD':
                try:
                    word_count[d[word]] += 1
                except KeyError:
                    continue
                start_index = max(0, i - window_size)
                end_index = min(len(sentence), i + window_size)
                for j in range(start_index, end_index):
                    if i == j:
                        continue
                    else:
                        target_word = sentence[j]
                        try:
                            pair_count_matrix[d[word], d[target_word]] += 1
                        except KeyError:
                            continue
    flag = 0
    for i in range (len(vocab)):
        for j in range(len(vocab)):
            if pair_count_matrix[i,j] >=min_cooccurence:
                flag = flag+1
                # print(pair_count_matrix[i,j])
            elif pair_count_matrix[i,j] <min_cooccurence:
                pair_count_matrix[i,j] = 0
    print('the number count of co-occurence of two words less than 2: {}'.format(flag))
            
    total_count = np.sum(word_count)
    word_count = word_count / total_count
    pair_count_matrix = pair_count_matrix / total_count
    
    pmi_matrix = np.zeros((len(vocab), len(vocab)), dtype=float)
    for i in range(len(vocab)):
        for j in range(len(vocab)):
            if word_count[i] * word_count[j] == 0:
                pmi_matrix[i, j]=0
            elif pair_count_matrix[i, j]==0:
                pmi_matrix[i, j]=0
            else:
                pmi_matrix[i, j] = np.log(
                    pair_count_matrix[i, j] / (word_count[i] * word_count[j]) 
                )
        
    pmi_matrix = np.nan_to_num(pmi_matrix)
    
    pmi_matrix = np.maximum(pmi_matrix, 0.0)

    edges_weights = [0.0]
    count = 1
    edges_mappings = np.zeros((len(vocab), len(vocab)), dtype=int)
    for i in range(len(vocab)):
        for j in range(len(vocab)):
            if pmi_matrix[i, j] != 0:
                edges_weights.append(pmi_matrix[i, j])
                edges_mappings[i, j] = count ###为了确定哪个位置有边
                count += 1
    print('the shape of edges_mapping: ', np.shape(edges_mappings))
    edges_weights = np.array(edges_weights)

    edges_weights = edges_weights.reshape(-1, 1)
    print('the shape of edges_weights: ',edges_weights.shape)
    edges_weights = torch.Tensor(edges_weights)
    
    return edges_weights, edges_mappings, count


if __name__ == "__main__":
    min_count=5
    data_root_path = 'data'
    vocab_root_path = 'data'
    cal_PMI(data_root_path, vocab_root_path, min_count, phase='train', window_size=6, min_cooccurence=2)
  
   