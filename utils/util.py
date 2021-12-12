import json
import os
import numpy as np
import torch
from torch.nn import Parameter
import pickle 
from PIL import Image
import shutil
import random
np.set_printoptions(suppress=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def store_preds_to_disk(tgts, preds, args):
    with open(os.path.join(args.savedir, "test_labels_pred.txt"), "w") as fw:
        fw.write("\n".join([str(x) for x in preds]))
    with open(os.path.join(args.savedir, "test_labels_gold.txt"), "w") as fw:
        fw.write("\n".join([str(x) for x in tgts]))
    with open(os.path.join(args.savedir, "test_labels.txt"), "w") as fw:
        fw.write(" ".join([str(l) for l in args.labels]))
def save_checkpoint(state, is_best, checkpoint_path, filename="checkpoint.pt"):
    filename = os.path.join(checkpoint_path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_path, "model_best.pt"))
def log_metrics(set_name, metrics, args, logger):
    if args.task_type == "multilabel":
        logger.info(
            "{}: Loss: {:.5f} | Macro F1 {:.5f} | Micro F1: {:.5f}".format(
                set_name, metrics["loss"], metrics["macro_f1"], metrics["micro_f1"]
            )
        )
    else:
        logger.info(
            "{}: Loss: {:.5f} | Acc: {:.5f}".format(
                set_name, metrics["loss"], metrics["acc"]
            )
        )


def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def load_checkpoint(model, path):
    best_checkpoint = torch.load(path)
    model.load_state_dict(best_checkpoint["state_dict"])

class Warp(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)

    def __str__(self):
        return self.__class__.__name__ + ' (size={size}, interpolation={interpolation})'.format(size=self.size,
                                                                                                interpolation=self.interpolation)

class MultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, 875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img):
        im_size = img.size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
        ret_img_group = crop_img_group.resize((self.input_size[0], self.input_size[1]), self.interpolation)
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset
    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret



def download_url(url, destination=None, progress_bar=True):
    """Download a URL to a local file.

    Parameters
    ----------
    url : str
        The URL to download.
    destination : str, None
        The destination of the file. If None is given the file is saved to a temporary directory.
    progress_bar : bool
        Whether to show a command-line progress bar while downloading.

    Returns
    -------
    filename : str
        The location of the downloaded file.

    Notes
    -----
    Progress bar use/example adapted from tqdm documentation: https://github.com/tqdm/tqdm
    """

    def my_hook(t):
        last_b = [0]

        def inner(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            if b > 0:
                t.update((b - last_b[0]) * bsize)
            last_b[0] = b

        return inner

    if progress_bar:
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            filename, _ = urlretrieve(url, filename=destination, reporthook=my_hook(t))
    else:
        filename, _ = urlretrieve(url, filename=destination)

def contac_jsons(data_root_path, save_root_path, dataset):
    ###将两个Json文件按某个key值拼接
    '''
    data_root_path = 'data/'
    save_root_path = 'data/'
    contac_jsons(data_root_path, save_root_path)
    '''
    data_splits =['train', 'val', 'test']
    for split_name in data_splits:
        print(split_name)
        mode = split_name
        path_1 = data_root_path +"{}/".format(dataset) + mode + ".jsonl" ###存储idxes, texts, images, labels
        path_2 = data_root_path +"{}/".format(dataset) + mode + "_anno.json"####存储idxes, objects
        save_path = save_root_path +"{}/".format(dataset) + mode + "_all_anno.json"
        with open(save_path, 'w') as fw:
            with open(path_1, 'r') as f1:
                for line in f1:
                    json_line = json.loads(line)
                    # print(type(json_line))
                    objects =[]
                    with open(path_2, 'r') as f2:
                        for line_ano in f2:
                            json_line_ano = json.loads(line_ano)
                            # print(type(json_line_ano))
                            if json_line['id']==json_line_ano['id']:
                                json_line['objects'] = json_line_ano['objects']
                                objects.append(json_line)
                    fw.write("%s\n" % json.dumps(json_line)) ###存储idxes, texts, images, labels, objects


def contac_object_and_place_jsons(text_data_root_path, object_data_root_path, place_data_root_path, save_root_path, dataset):
    '''
    ###将object/'split(train/dev/test)'_all_anno.json 与 place/'split(train/dev/test)'_all_anno.json 
    # 文件按某个key值拼接
    text_data_path ='data/dataset/text_data/split-stemmed.txt'
    object_data_path = 'data/dataset/object/split_all_anno.json'
    place_data_path = 'data/dataset/place/split_all_anno.json'
    save_path = 'data/dataset/all_anno_json/split_all_anno.json'
    dataset='MVSA_simple' or 'MVSA_multiple' or 'tumblr'
    '''
    data_splits =['train']
    for split_name in data_splits:
        print(split_name)
        mode = split_name
        ###id label text
        text_data_path = text_data_root_path + "{}/".format(dataset) + 'text_data/'+ mode + "-stemmed.txt"
        with open(text_data_path, 'r') as ft:
            text_lines = ft.read().split('\n')
        ###存储idxes, texts, images, labels, objects
        object_data_path = object_data_root_path +"{}/".format(dataset) + 'object/'+ mode + "_all_anno.json" 
        ###存储idxes, texts, images, labels, places
        place_data_path = place_data_root_path +"{}/".format(dataset) + 'place/'+ mode + "_place_all_anno.json"
        save_path = save_root_path +"{}/all_anno_json/".format(dataset) + mode + "_all_anno.json"
        with open(save_path, 'w') as fw:
            with open(object_data_path, 'r') as f1:
                for line in f1:
                    json_line = json.loads(line)
                    for i in range(len(text_lines)):
                        text_id, label, text = text_lines[i].split('\t')
                        if json_line['id'] == text_id:
                            # print(json_line['id'], text_id)
                            json_line['text'] = text
                            # print(type(json_line))
                            places =[]
                            with open(place_data_path, 'r') as f2:
                                for line_ano in f2:
                                    json_line_ano = json.loads(line_ano)
                                    # print(type(json_line_ano))
                                    if json_line['id']==json_line_ano['id']:
                                        json_line['places'] = json_line_ano['places']
                                        places.append(json_line)
                            fw.write("%s\n" % json.dumps(json_line)) ###存储idxes, texts, images, labels, objects

def remove_short(json_old_data_root_path, save_data_root_path, dataset,split):
    json_old_data_path = os.path.join(json_old_data_root_path, dataset, 'all_anno_json', "{}_all_anno.json".format(split))
    print(json_old_data_path)
    save_data_path_ = os.path.join(save_data_root_path, dataset, 'all_anno_json') 
    if not os.path.exists(save_data_path_):
        os.makedirs(save_data_path_)
    save_data_path = os.path.join(save_data_path_, "{}_all_anno.json".format(split)) 
    
    all_data = []
   
    with open(json_old_data_path, 'r') as f:
        for line in f:
            print(line)
            json_line = json.loads(line)
            print(json_line)
            id = json_line['id']
            text = json_line['text']
            text_list = text.split(' ')
            print(len(text_list))
            if len(text_list) < 5:
                continue
            else:
                all_data.append(json_line)
   
    with open(save_data_path, 'w') as fw:
        for i in range(len(all_data)):
            fw.write("%s\n" % json.dumps(all_data[i]))
 
    print(len(all_data))


def up_sampling(train_data_path):
    neutral_list = []
    with open(train_data_path, 'r') as f:
        for line in f:
            json_line = json.loads(line)
            if json_line['label'] == 'neutral':
                neutral_list.append(json_line)
        f.close()
    with open(train_data_path, 'a') as fw:
        for i in neutral_list:
            fw.write("%s\n" % json.dumps(i)) ###存储idxes, texts, images, labels, objects

    return neutral_list

def calculate_label_num(data_path):
    pos_len = 0
    neg_len =0
    neu_len = 0
    with open(data_path, 'r') as f:
        for line in f:
            json_line = json.loads(line)
            if json_line['label'] == 'neutral':
                neu_len = neg_len+1
            elif json_line['label'] == 'negative':
                neg_len = neg_len +1
            elif json_line['label'] == 'positive':
                pos_len = pos_len+1
    return neu_len, pos_len, neg_len



def return_objects(data_root_path, mode, dataset):
    ###读取json文件中的objects为之后的Adj建立做准备
    objects = []    
    path = os.path.join(data_root_path, dataset, "{}_anno.json".format(mode))
    with open(path, 'r') as f:
        for line in f:
            json_line = json.loads(line)
            _object = list(set(json_line['objects']))
            objects.append(_object)
    return objects

def generate_nums(objects, num_classes):
    nums = np.zeros(num_classes) ##[1,80] 每个位置代表出现的总次数
    # print(nums)
    # adj = np.zeros(80,80)
    for i in range(len(objects)):
        if len(objects[i])!=0:
            # print(objects[i])
            for j in objects[i]:
                nums[j] = nums[j]+1
    # print(nums)
    return nums
def generate_Adj(objects, num_classes):
    ###根据object共现次数，填充Adj矩阵
    Adj = np.zeros((num_classes, num_classes))
    for i  in range(len(objects)):
        if len(objects[i]) != 0:
            for j, element in enumerate(objects[i]):
                for k , element_ano in enumerate(objects[i]):
                    if element !=  element_ano:
                        Adj[element][element_ano] = Adj[element][element_ano]+1
    return Adj


def get_Adj(data_root_path, data_splits, num_classes, dataset):
    ###获得整个数据集{train, val, test}的Adj
    all_nums = np.zeros(num_classes)
    all_Adj = np.zeros((num_classes, num_classes))
    for split_name in data_splits:
        objects = return_objects(data_root_path, split_name, dataset)
        nums = generate_nums(objects, num_classes)
        nums = np.array(nums)
        all_nums = all_nums + nums
        Adj =  generate_Adj(objects, num_classes)
        Adj = np.array(Adj)
        all_Adj = all_Adj +Adj
    for i in range(len(all_nums)):
        if all_nums[i] ==0:
            ###为避免0/0情况出现，将出现0次的物体赋值为1
            all_nums[i]=1
    result={}
    result['nums'] = all_nums
    result['adj'] = all_Adj
    with open("data/{}_adj.pkl".format(dataset), 'wb') as fo:
        pickle.dump(result, fo)
    return all_nums, all_Adj

def gen_A(num_classes, t, adj_file, gama):
    ###使用参数t,来过滤一部分长尾噪音，即共现次数少的，认为没有共现性，
    # 但是我的数据集太小，所以把t值设的小一些，需要实验验证
    ### t: we use the threshold t to filter noisy edges,
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    print('======================_adj======================')
    print(np.shape(_adj))
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0 
    _adj[_adj >= t] = 1
    _adj = _adj * gama / (_adj.sum(0, keepdims=True) + 1e-6) ###0.2为论文中的p值
    _adj = _adj + (1-gama) * np.identity(num_classes, np.int)###加入自连接，对角线元素为1, (1-0.2)可以考虑乘不乘
    return _adj, _nums
# def gen_A(num_classes, t):
#     ###t: we use the threshold t to filter noisy edges,
#     # result = pickle.load(open(adj_file, 'rb'))
#     # _adj = result['adj']
#     # print('------------------------_adj--------------')
#     # print(_adj[0][:])
#     # _nums = result['nums']###[80] ###'nums'是该object出现的次数
#     data_root_path = '/data/yxc/code/MVSA/MVSA_simple/data/'
#     data_splits =['train', 'val', 'test']
#     # data_splits =['train']
#     all_nums, all_Adj = get_Adj(data_root_path, data_splits)
#     _adj = all_Adj
#     _nums = all_nums

#     _nums = _nums[:, np.newaxis]##[80,1]
#     _adj = _adj / _nums
#     _adj[_adj < t] = 0
#     _adj[_adj >= t] = 1
#     print(_adj[0][:])
#     _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
#     _adj = _adj + np.identity(num_classes, np.int)
#     return _adj
def gen_adj(A):
    ###D为度矩阵，从而通过节点的度对特征表征进行归一化
    D = torch.pow(A.sum(1).float(), -0.5) ###A.sum(1):对A张量，在dim=1（列）的维度上求和，1/(根号下(A.sum(1)))
    D = torch.diag(D)##建立对角矩阵 
    adj = torch.matmul(torch.matmul(A, D).t(), D)###((AXD)^T)XD
    return adj


if __name__ == "__main__":  
    # data_root_path = 'data/MVSA_simple/'
    # data_splits =['train', 'val', 'test']
    # # # data_splits =['train']
    # all_nums, all_Adj = get_Adj(data_root_path, data_splits)
    # # print(all_nums)
    # # print(all_Adj[1][:])
    # num_classes = 80
    # t= 0.1
    # adj_file = 'data/MVSA_simple_adj.pkl'
    # _adj,_nums = gen_A(num_classes, t, adj_file)
    # print(_nums[2])
    # print(_adj[2][:])
#     A=Parameter(torch.from_numpy(_adj).float())
#     adj = gen_adj(A).detach()

#     for i in range(len(_nums)):
#         print(_nums[i])
#         print(_adj[i][:])
#         # print(_nums)remove_short(json_old_data_root_path, save_data_root_path, dataset,split)
    # print(_adj[48][:])
    # # print(_nums)
    # print(adj[2][:])
    # train_data_path = 'data/MVSA_simple/train_all_anno.json'
    # up_sampling(train_data_path)
    # text_data_root_path = 'data/'
    # object_data_root_path = 'data/'
    # place_data_root_path = 'data/'
    # save_root_path = 'data/'
    # dataset = 'MVSA_multiple'
    # contac_object_and_place_jsons(text_data_root_path,  object_data_root_path, place_data_root_path, save_root_path, dataset)

    json_old_data_root_path = 'data'
    save_data_root_path = 'data'
    dataset = 'MVSA_multiple'
    split = 'test'
    remove_short(json_old_data_root_path, save_data_root_path, dataset,split)

    # contac_jsons(data_root_path, save_root_path, dataset)
    # neu_len, pos_len, neg_len = calculate_label_num(train_data_path)
    # print( neu_len, pos_len, neg_len)
        
