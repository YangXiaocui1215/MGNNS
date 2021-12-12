
##CUDA_VISIBLE_DEVICES=3 python3 MVSA_Multi_GCN_Multihead_Att.py  --image-size 448 --batch-size 4 -e --text_min_count 5 --ngram 4 --window_size 4 --epochs 10 --lr 5e-5 --object_t_value 0.3 --place_t_value 0.3
import argparse
from models.Multi_GCN_Multihead_att_new import multi_gcn_multihead_att_model
from utils.Multi_GCN_Co_att_dataset_new import Tumblr_Dataset
from utils.util import *
from utils.vocab_new import get_vocab_list
from engine.Multi_GCN_Multihead_Att_engine import MultiClassEngine,GCNMultiClassEngine
import torch.nn as nn
# from Dual_GCNResnet_engine_gradient_accumulation import MultiClassEngine, GCNMultiClassEngine ###梯度累积

parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument("--dataset", type=str, default="tumblr")
parser.add_argument('--data_root_path', type=str, default='data', metavar='DIR', help='path to dataset (e.g. data')
parser.add_argument('--bidirectional', type=bool, default=True, help='if is the bidirectional-GRU')
parser.add_argument('--hidden_size', type=int, default=150, help='dimension of gru hidden states')
parser.add_argument('--emb_size', type=int, default=300, help='dimension of glove embedding')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers in GRU')
parser.add_argument('-dropout', type=float, default=0.5)
parser.add_argument('-emb_type', type=str, choices=['random', 'glove', 'glove200d', 'glove300d', 'fasttext300d'], default='glove', help='')
parser.add_argument('--stack_num', type=int, default=2, help='number of stack in multihead attention')
parser.add_argument('--n_head', type=int, default=4, help='number of the head in multihead attention')
parser.add_argument('--d_kv', type=int, default=128, help='the dim of each head in multihead attention')
parser.add_argument('--is_regu', type=bool, default=False, help='add is_regu argument and diff_outputs function to support the diff head loss')

parser.add_argument('--text_min_count', type=int, default=5,
                    help='if the word count>=text_min_count, the word is saved in vocab')
parser.add_argument('--window_size', type=int, default=6,
                    help='MVSA_simple: 5,6,7; MVSA_multiple: 10,11')
parser.add_argument('--ngram', type=int, default=4,
                    help='similar to windowsize, if ngram=4, 则以当前单词为中心点，将当前单词的前4个词与后4个词建立边')
parser.add_argument('--min_cooccurence', type=int, default=2,
                    help='if the cooccurence of two words>=min_cooccurence, the cooccurence is saved in vocab')

parser.add_argument('--image-size', '-i', default=448, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[10], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--device_ids', default=[1], type=int,
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--save_experiment_result_path', default='result/experiment_result', type=str,
                    help='path to save the test_dataset experiment result')###由于后面还加参数，故该文件名没有写全
parser.add_argument('--save_pred_result_path', default='result/pred_result', type=str,
                    help='path to save the test_dataset pred result')###由于后面还加参数，故该文件名没有写全
parser.add_argument('--model_name', type=str, default='Multi_GCN_Multihead_Att_new')
parser.add_argument('--save_model_path', default='checkpoint', type=str)
parser.add_argument('--object_t_value', type=float, default=0.4, help=' t: we use the threshold t to filter noisy edges,') 
######使用参数t,来过滤一部分长尾噪音，即共现次数少的，认为没有共现性，
# 但是我的数据集太小，所以把t值设的小一些，需要实验验证
parser.add_argument('--place_t_value', type=float, default=0.3, help=' t: we use the threshold t to filter noisy edges,') 
parser.add_argument('--num_labels', type=int, default=7, help = 'the number of labels')
parser.add_argument('--object_num_classes', type=int, default=80, help = 'the number of object catagaries in object_Adj')###邻接矩阵中object类别数目
parser.add_argument('--place_num_classes', type=int, default=365, help = 'the number of place catagaries in place_Adj')###邻接矩阵中place类别数目
parser.add_argument('--accumulation_steps', type=int, default=8, help='accumulation gradientto expand the batch_size') 
parser.add_argument("--fp16",action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",)
parser.add_argument("--fp16_opt_level",type=str,default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",)

def main_MVSA():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    emb_path = os.path.join(args.data_root_path, 'glove_embedding', 'glove_embedding_{}.pkl'.format(args.text_min_count))
    if os.path.exists(emb_path):
        print('The glove_embedding has built!!')
    else:
        print('Not found the embedding file: {}'.format(emb_path))
    data_path = os.path.join(args.data_root_path, 'all_anno_json')
    vocab_path = os.path.join(args.data_root_path, 'vocab')
    vocab = get_vocab_list(args.data_root_path, 
                           args.data_root_path,
                           args.text_min_count)
    vocab_size = len(vocab)

    opt = {'emb_path': emb_path,
           'bidirectional': args.bidirectional, 
           'hidden_size': args.hidden_size,
           'emb_size': args.emb_size,
           'num_layers': args.num_layers,
           'dropout': args.dropout,
           'emb_type': args.emb_type,
           'vocab_size': vocab_size,
           'stack_num': args.stack_num,
           'n_head': args.n_head,
           'd_kv': args.d_kv,
           'is_regu': args.is_regu
           }
    print('---------opt-------------')
    print(opt['emb_path'])


    ###load dataset
    train_dataset = Tumblr_Dataset(root=args.data_root_path,
                                 dataset=args.dataset,
                                 text_min_count=args.text_min_count,
                                 vocab=None,
                                 transform=None,
                                 phase='train',
                                 object_inp_name='data/glove/object_glove_word2vec.pkl', 
                                 place_inp_name='data/glove/place_glove_word2vec.pkl')
    val_dataset = Tumblr_Dataset(root=args.data_root_path,
                                 dataset=args.dataset,
                                 text_min_count=args.text_min_count,
                                 vocab=None,
                                 transform=None,
                                 phase='val',
                                 object_inp_name='data/glove/object_glove_word2vec.pkl', 
                                 place_inp_name='data/glove/place_glove_word2vec.pkl')
    test_dataset =  Tumblr_Dataset(root=args.data_root_path,
                                 dataset=args.dataset,
                                 text_min_count=args.text_min_count,
                                 vocab=None,
                                 transform=None,
                                 phase='test',
                                 object_inp_name='data/glove/object_glove_word2vec.pkl', 
                                 place_inp_name='data/glove/place_glove_word2vec.pkl')

    ###build model
    model = multi_gcn_multihead_att_model(opt=opt,
                            num_labels=args.num_labels, 
                            object_num_classes=args.object_num_classes, place_num_classes=args.place_num_classes,
                            object_t=args.object_t_value, place_t=args.place_t_value,
                            data_root_path=args.data_root_path, vocab_root_path=args.data_root_path, 
                            text_min_count=args.text_min_count,
                            window_size=args.window_size,
                            ngram=args.ngram, 
                            min_cooccurence=args.min_cooccurence, 
                            text_dropout=0.5,
                            pretrained=True, 
                            object_adj_file='data/adj/tumblr_objects_adj.pkl', 
                            place_adj_file='data/adj/tumblr_resnet50_places_adj.pkl',
                            in_channel=300)
    # print(model)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss()

    # define optimizer
    optimizer = torch.optim.Adam(model.get_config_optim(args.lr, args.lrp), 
                                lr=args.lr, 
                                # momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                )

    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 
             'object_num_classes':args.object_num_classes, 'place_num_classes':args.place_num_classes, 
             'model_name':args.model_name}

    state['save_experiment_result_path'] = os.path.join(args.save_experiment_result_path, args.model_name)
    os.makedirs(state['save_experiment_result_path'], exist_ok=True)

    state['save_pred_result_path'] = os.path.join(args.save_pred_result_path, args.model_name)
    os.makedirs(state['save_pred_result_path'], exist_ok=True)

    state['save_model_path'] = os.path.join(args.save_model_path, args.model_name)
    os.makedirs(state['save_model_path'], exist_ok=True)
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr

    state['text_min_count'] = args.text_min_count
    state['ngram'] = args.ngram
    state['window_size'] = args.window_size

    state['object_t_value'] = args.object_t_value
    state['place_t_value'] = args.place_t_value
    state['accumulation_steps']= args.accumulation_steps
    state['fp16'] = args.fp16
    state['fp16_opt_level'] = args.fp16_opt_level
    # state['device_ids'] = args.device_ids
    if args.evaluate:
        state['evaluate'] = True
    ###梯度累积
    engine = GCNMultiClassEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, test_dataset, optimizer)

if __name__ == '__main__':
    main_MVSA()
