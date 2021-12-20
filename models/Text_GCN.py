import dgl
import torch
import torch.nn.functional as F
import numpy as np
import word2vec
import warnings
warnings.filterwarnings("ignore")

def gcn_msg(edge):
    ##Edge UDFs 需要传入一个 edge 参数，其中 edge 有三个属性：src、dst、data，分别对应源节点特征、目标节点特征和边特征。
    return {'m': edge.src['h'], 'w': edge.data['w']}


def gcn_reduce(node):
    '''
    对于每个节点来说，可能过会收到很多个源节点传过来的消息，所以可以将这些消息存储在邮箱中（mailbox）。
    我们那再来定义一个聚合（Reduce）函数。
    消息传递完后，每个节点都要处理下他们的“信箱”（mailbox），Reduce 函数的作用就是用来处理节点“信箱”的消息的。
    Reduce 函数是一个 Node UDFs。
    Node UDFs 接收一个 node 的参数，并且 node 有两个属性 data 和 mailbox，分别为节点的特征和用来接收信息的“信箱”。
    '''
    w = node.mailbox['w']

    new_hidden = torch.mul(w, node.mailbox['m'])

    new_hidden,_ = torch.max(new_hidden, 1)

    node_eta = torch.sigmoid(node.data['eta'])
    # node_eta = F.leaky_relu(node.data['eta'])

    # new_hidden = node_eta * node.data['h'] + (1 - node_eta) * new_hidden
    # print(new_hidden.shape)

    return {'h': new_hidden}

class Model(torch.nn.Module):
    def __init__(self,
                 class_num,
                 hidden_size_node,
                 vocab,
                 n_gram,
                 drop_out,
                 edges_num,
                 edges_matrix,
                 max_length=100,
                 trainable_edges=True,
                 pmi=None,
                 cuda=True,
                 is_padding=True
                 ):
        super(Model, self).__init__()

        self.is_cuda = cuda
        self.is_padding = is_padding
        self.vocab = vocab
        # print(len(vocab))
        print("-------------------edge initialization---------------")
        self.seq_edge_w = torch.nn.Embedding(edges_num, 1)
        print(edges_num)
        print(pmi.shape)

        self.node_hidden = torch.nn.Embedding(len(vocab), hidden_size_node)
       
        # self.seq_edge_w = torch.nn.Embedding.from_pretrained(pmi, freeze=False)
       
        self.edges_num = edges_num
        if trainable_edges:
            self.seq_edge_w = torch.nn.Embedding.from_pretrained(torch.ones(edges_num, 1), freeze=False)
            print("-------------------node initialization by 1--------")

        else:
            self.seq_edge_w = torch.nn.Embedding.from_pretrained(pmi, freeze=False)
            print("-------------------node initialization by pmi--------")
        self.hidden_size_node = hidden_size_node
        print('---------------2-----------------------')
        self.node_hidden.weight.data.copy_(torch.tensor(self.load_word2vec('glove/glove.6B.300d.txt')))
        # self.node_hidden.weight.data.copy_(torch.tensor(self.load_word2vec('glove/glove.twitter.27B.200d.txt')))
        print('--------------------------3------------------')
        self.node_hidden.weight.requires_grad = True

        self.len_vocab = len(vocab)

        self.ngram = n_gram

        self.d = dict(zip(self.vocab, range(len(self.vocab))))

        self.max_length = max_length

        self.edges_matrix = edges_matrix

        self.dropout = torch.nn.Dropout(p=drop_out)

        self.activation = torch.nn.ReLU()

        self.Linear = torch.nn.Linear(hidden_size_node, class_num, bias=True)

    def word2id(self, word):
        try:
            result = self.d[word]
        except KeyError:
            result = self.d['UNK']

        return result

    def load_word2vec(self, word2vec_file):
        model = word2vec.load(word2vec_file)

        embedding_matrix = []

        for word in self.vocab:
            try:
#                print(word)
                embedding_matrix.append(model[word])
            except KeyError:
                # print(word)
                embedding_matrix.append(model['the'])

        embedding_matrix = np.array(embedding_matrix)
        print('the shape of  embedding_matrix is: ',np.shape(embedding_matrix))

        return embedding_matrix

    def add_all_edges(self, doc_ids: list, old_to_new: dict):
        edges = []
        old_edge_id = []

        local_vocab = list(set(doc_ids))

        for i, src_word_old in enumerate(local_vocab):
            src = old_to_new[src_word_old]
            for dst_word_old in local_vocab[i:]:
                dst = old_to_new[dst_word_old]
                edges.append([src, dst])
                old_edge_id.append(self.edges_matrix[src_word_old, dst_word_old])

            # self circle
            edges.append([src, src])
            old_edge_id.append(self.edges_matrix[src_word_old, src_word_old])

        return edges, old_edge_id

    def add_seq_edges(self, doc_ids: list, old_to_new: dict):
        edges = []
        old_edge_id = []
        new_doc_ids = []
        ####去除text_padding：0
        for id_ in doc_ids:
            if id_!=0:
                new_doc_ids.append(id_)
        doc_ids = new_doc_ids        
        for index, src_word_old in enumerate(doc_ids):
            src = old_to_new[src_word_old]
            for i in range(max(0, index - self.ngram), min(index + self.ngram + 1, len(doc_ids))):
                dst_word_old = doc_ids[i]
                dst = old_to_new[dst_word_old]

                # - first connect the new sub_graph
                edges.append([src, dst])
                # - then get the hidden from parent_graph
                old_edge_id.append(self.edges_matrix[src_word_old, dst_word_old])

            # self circle
            edges.append([src, src])
            old_edge_id.append(self.edges_matrix[src_word_old, src_word_old])

        return edges, old_edge_id

    def seq_to_graph(self, doc_ids: list) -> dgl.DGLGraph():
        if len(doc_ids) > self.max_length:
            doc_ids = doc_ids[:self.max_length]

        local_vocab = set(doc_ids)

        old_to_new = dict(zip(local_vocab, range(len(local_vocab))))

        if self.is_cuda:
            local_vocab = torch.tensor(list(local_vocab)).cuda()
        else:
            local_vocab = torch.tensor(list(local_vocab))

        sub_graph = dgl.DGLGraph().to('cuda:0')

        sub_graph.add_nodes(len(local_vocab))
        local_node_hidden = self.node_hidden(local_vocab)###torch.Size([the sentence length, 200])
        # print('-----the local_node_hidden.size is {}------------'.format(local_node_hidden.size()))

        sub_graph.ndata['h'] = local_node_hidden

        seq_edges, seq_old_edges_id = self.add_seq_edges(doc_ids, old_to_new)

        edges, old_edge_id = [], []
        # edges = []

        edges.extend(seq_edges)

        old_edge_id.extend(seq_old_edges_id)

        if self.is_cuda:
            old_edge_id = torch.LongTensor(old_edge_id).cuda()
        else:
            old_edge_id = torch.LongTensor(old_edge_id)

        srcs, dsts = zip(*edges) ###srcs:源节点, dsts：目标节点 
        sub_graph.add_edges(srcs, dsts)
        try:
            seq_edges_w = self.seq_edge_w(old_edge_id)
        except RuntimeError:
            print(old_edge_id)
        sub_graph.edata['w'] = seq_edges_w

        return sub_graph

    def forward(self, doc_ids, is_20ng=None):
        ###doc_ids就是文本内容
        # print('-------------------doc_ids----------------------')
        # # print(doc_ids)
        # print(type(doc_ids[0]))
        # # print(len(doc_ids))
        # new_doc_ids = []
        # for i in range(len(doc_ids)):
        #     new_ids = []
        #     # print('=====================doc_ids{}=========================='.format(i))
        #     # print(doc_ids[i])
        #     ids = doc_ids[i]
        #     for j in range(len(ids)):
        #         if ids[j]!=0:
        #             new_ids.append(ids[j])
        #     new_doc_ids.append(new_ids)

        # sub_graphs = [self.seq_to_graph(torch.masked_select(doc, doc!=0)) for doc in doc_ids]
        ###将输入的tensor张量，转化为list
        doc_ids = doc_ids.cpu().numpy().tolist()

        sub_graphs = [self.seq_to_graph(doc) for doc in doc_ids]

        batch_graph = dgl.batch(sub_graphs)
        before_node_embedding = batch_graph.ndata['h']
        # print('-----------------------before updating----------------------')
        # print(before_node_embedding.size())
        # print(before_node_embedding)

        batch_graph.update_all(
            message_func=dgl.function.src_mul_edge('h', 'w', 'weighted_message'),
            ###h为源特征，w为目标特征域，weighted_message 为output message field.
            ###通过节点特征h与边特征w的mul运算，得weighted_message

            reduce_func=dgl.function.max('weighted_message', 'h') ##聚合邻居信息
            ##内置的reduce功能最大可聚合消息，聚合后赋值到h。
        )
        # for nodes in batch_graph.nodes():
        #     print(nodes)
            # print(nodes.ndata['h'])
        after_node_embedding = batch_graph.ndata['h']
        # print('-----------------------after updating----------------------')
        # print(batch_graph)
        # print(after_node_embedding.size())
        # print(after_node_embedding)
        node_eta = torch.nn.Parameter(torch.FloatTensor(1), requires_grad = True).cuda()
        node_eta.data.fill_(0)
        # print('====================node_eta is: {}=========='.format(node_eta))
        new_node_embedding = node_eta * before_node_embedding + (1 - node_eta) * after_node_embedding
        batch_graph.ndata['h'] = new_node_embedding
        # print('-----------------------after weighting----------------------')
        # print(batch_graph.ndata['h'].size())
        # print(batch_graph.ndata['h'])


        h1 = dgl.sum_nodes(batch_graph, feat='h')

        drop1 = self.dropout(h1)
        act1 = self.activation(drop1)

        # l = self.Linear(act1)

        return act1
        # print(batch_graph)
        h1 = dgl.sum_nodes(batch_graph, feat='h')
        # print('--------------h1.size-----------------------------------------')
        # # print(h1.size())
        # w1 = dgl.sum_edges(batch_graph, feat='h')
        # print('--------------w1.size-----------------------------------------')
        # print(w1.size())

        drop1 = self.dropout(h1)
        act1 = self.activation(drop1)

        # l = self.Linear(act1)

        return act1
