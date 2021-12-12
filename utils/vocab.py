import os
import json
import csv
"""
根据train data 建立vocab
"""

def get_vocab_list(data_root_path, vocab_root_path, text_min_count):
    try:
        vocab = get_vocab(vocab_root_path, text_min_count)
    except FileNotFoundError:
        train_all_text = get_content(data_root_path)
        vocab = build_vocab(vocab_root_path, train_all_text, text_min_count)
    return vocab


def get_content(data_root_path):
    all_text =[]
    text_data_path = os.path.join(data_root_path, 'all_anno_json_new', 'train_all_anno.json')
    with open(text_data_path, 'r') as f:
        for line in f:
            json_line = json.loads(line)
            text = json_line['text']
            all_text.append(text)
    return all_text

def get_vocab(vocab_root_path, text_min_count):
    with open(os.path.join(vocab_root_path, 'vocab_new', 'vocab-' + str(text_min_count)+'.txt')) as f:
        print('------------------------geting vocab-----------------------')
        vocab = f.read()
        vocab = vocab.split('\n')
        print('the length of vocab is: ', len(vocab))
    return vocab

def build_vocab(vocab_root_path, train_all_text, text_min_count):
    print('------------------building vocab,使用train数据---------------------')
    vocab = []
    for text in train_all_text:
        words = text.split(' ')
        for word in words:
            if word not in vocab:
                vocab.append(word)

   
    freq = dict(zip(vocab, [0 for i in range(len(vocab))]))
    for text in train_all_text:
        words = text.split(' ')
        for word in words:
            freq[word] += 1
            
    if not os.path.exists(os.path.join(vocab_root_path, 'freq.csv')):
        print('-------------no freq.csv, so save it----------')
        with open(os.path.join(vocab_root_path, 'vocab_new', 'freq.csv'), 'w') as f:
            writer = csv.writer(f)
            results = list(zip(freq.keys(), freq.values()))
            writer.writerows(results)

    results = []
    for word in freq.keys():
        if freq[word] < text_min_count:
            continue
        else:
            results.append(word)

    results.insert(0, 'PAD')
    results.insert(1, 'UNK')
    with open(os.path.join(vocab_root_path, 'vocab_new','vocab-' + str(text_min_count)+'.txt'), 'w') as f:
        f.write('\n'.join(results))

    return results ###vocab_list

if __name__ == '__main__':
    data_root_path = vocab_root_path = 'data'
    text_min_count = 6

    vocab = get_vocab_list(data_root_path, vocab_root_path, text_min_count)
    

