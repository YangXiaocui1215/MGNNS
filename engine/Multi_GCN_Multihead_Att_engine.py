'''
运型环境：yxc_py36, 运行指令：
CUDA_VISIBLE_DEVICES=3 python3 MVSA_gcn.py --image-size 448 --batch-size 32 -e 
'''

import os
import shutil
import time
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet as tnt
import torchvision.transforms as transforms
import torch.nn as nn
from utils.util import *
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report
from tqdm import tqdm
from apex import amp

tqdm.monitor_interval = 0
class Engine(object):
    def __init__(self, state={}):
        self.state = state
        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()

        if self._state('image_size') is None:
            self.state['image_size'] = 224

        if self._state('batch_size') is None:
            self.state['batch_size'] = 16

        if self._state('workers') is None:
            self.state['workers'] = 25

        if self._state('device_ids') is None:
            self.state['device_ids'] = [1]

        if self._state('evaluate') is None:
            self.state['evaluate'] = False

        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0

        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 90

        if self._state('object_t_value') is None:
            self.state['object_t_value'] = 0.4
        if self._state('place_t_value') is None:
            self.state['place_t_value'] = 0.4

        if self._state('epoch_step') is None:
            self.state['epoch_step'] = []
        ###定义一个epoch_acc，初始化为0
        if self._state('epoch_acc') is None:
            self.state['epoch_acc'] = 0 ###accuracy_score(tgts, preds)
        if self._state('batch_acc_list') is None:
            self.state['batch_acc_list'] = [] ###accuracy_score(tgts, preds)

        ###定义epoch_f1,初始化为0, f1_score(tgts, preds, averge='micro')
        ###Calculate metrics globally by counting the total true positives, false negatives and false positives.
        if self._state('epoch_micro_f1') is None:
            self.state['epoch_micro_f1'] = 0 ###f1_score(tgts, preds)
        if self._state('batch_micro_f1_list') is None:
            self.state['batch_micro_f1_list'] = [] ###f1_score(tgts, preds)

        ###f1_score(tgts, preds, averge='macro')
        ###Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
        if self._state('epoch_macro_f1') is None:
            self.state['epoch_macro_f1'] = 0 ###f1_score(tgts, preds)
        if self._state('batch_macro_f1_list') is None:
            self.state['batch_macro_f1_list'] = [] ###f1_score(tgts, preds)

         ###f1_score(tgts, preds, averge='weighted')    
         ###Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). 
         # This alters ‘macro’ to account for label imbalance; 
         # it can result in an F-score that is not between precision and recall.
        if self._state('epoch_weighted_f1') is None:
            self.state['epoch_weighted_f1'] = 0 ###f1_score(tgts, preds)
        if self._state('batch_weighted_f1_list') is None:
            self.state['batch_weighted_f1_list'] = [] ###f1_score(tgts, preds)

        if self._state('id_list') is None:
            self.state['id_list'] = [] 
        ###记录预测后的结果
        if self._state('pred_list') is None:
            self.state['pred_list'] = [] 
        ###记录标签的结果，用于之后将真实标签与预测标签进行比较
        if self._state('target_list') is None:
            self.state['target_list'] = [] 
        
        if self._state('fp16') is None:
            state['fp16']  = True
        if self._state('fp16_opt_level') is None:
            state['fp16_opt_level']  = 'O1'

        # meters
        #AverageValueMeter测量并返回添加到其中的任何数字集合的平均值和标准差,
        self.state['meter_loss'] = tnt.meter.AverageValueMeter()
        # time measure
        self.state['batch_time'] = tnt.meter.AverageValueMeter()
        self.state['data_time'] = tnt.meter.AverageValueMeter()
        # display parameters
        if self._state('use_pb') is None:
            self.state['use_pb'] = True
        if self._state('print_freq') is None:
            self.state['print_freq'] = 10

    def _state(self, name):
        if name in self.state:
            return self.state[name]

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()

        self.state['batch_acc_list'].clear()

        self.state['batch_micro_f1_list'].clear()
        self.state['batch_macro_f1_list'].clear()
        self.state['batch_weighted_f1_list'].clear()

        self.state['id_list'].clear()
        self.state['target_list'].clear()
        self.state['pred_list'].clear()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        loss = self.state['meter_loss'].value()[0] ###返回的是平均值
        ##acc
        self.state['epoch_acc'] = sum(self.state['batch_acc_list'])/len(data_loader) ###计算一个epoch的平均acc
        acc = self.state['epoch_acc']
        ###micro_f1
        self.state['epoch_micro_f1'] = sum(self.state['batch_micro_f1_list'])/len(data_loader) ###计算一个epoch的平均f1
        micro_f1 = self.state['epoch_micro_f1']
        ###macro_f1
        self.state['epoch_macro_f1'] = sum(self.state['batch_macro_f1_list'])/len(data_loader) ###计算一个epoch的平均f1
        macro_f1 = self.state['epoch_macro_f1']
        ##weighted_f1
        self.state['epoch_weighted_f1'] = sum(self.state['batch_weighted_f1_list'])/len(data_loader) ###计算一个epoch的平均f1
        weighted_f1 = self.state['epoch_weighted_f1']

        if display:
            if training:
                print('Epoch: [{0}]\t'
                      'Loss {loss:.4f}\t'
                      'Acc {acc:.4f}\t'
                      'Micro_F1-score {micro_f1:.4f}\t'
                      'Macro_F1-score {macro_f1:.4f}\t'
                      'Weighted_F1-score {weighted_f1:.4f}'.format(self.state['epoch'], 
                                                loss=loss, 
                                                acc=acc,
                                                micro_f1=micro_f1,
                                                macro_f1=macro_f1,
                                                weighted_f1=weighted_f1
                                                ))
            else:
                print('Val: \t '
                      'Loss {loss:.4f}\t'
                      'Acc {acc:.4f}\t'
                      'Micro_F1-score {micro_f1:.4f}\t'
                      'Macro_F1-score {macro_f1:.4f}\t'
                      'Weighted_F1-score {weighted_f1:.4f}'.format(loss=loss, 
                                                 acc=acc,
                                                 micro_f1=micro_f1,
                                                macro_f1=macro_f1,
                                                weighted_f1=weighted_f1))
        return loss, acc, micro_f1, macro_f1, weighted_f1, self.state['id_list'], self.state['target_list'], self.state['pred_list']

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        pass

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        # record loss
        self.state['loss_batch'] = self.state['loss'].item()
        self.state['meter_loss'].add(self.state['loss_batch']) ## 把每个batch的损失加入，计算平均损失
        # record acc,每个batch的acc
        self.state['batch_acc'] = self.state['acc']
        self.state['batch_acc_list'].append(self.state['batch_acc'])
        ##record micro_f1,每个batch的f1
        self.state['batch_micro_f1'] = self.state['micro_f1']
        self.state['batch_micro_f1_list'].append(self.state['batch_micro_f1'])
         ##record macro_f1,每个batch的f1
        self.state['batch_macro_f1'] = self.state['macro_f1']
        self.state['batch_macro_f1_list'].append(self.state['batch_macro_f1'])
         ##record weighted_f1,每个batch的f1
        self.state['batch_weighted_f1'] = self.state['weighted_f1']
        self.state['batch_weighted_f1_list'].append(self.state['batch_weighted_f1'])
        ###将每batch的id, target, pred 保存至列表中
        self.state['id_list'] = self.state['id_list'] + self.state['id']
        self.state['pred_list'] = self.state['pred_list'] + self.state['pred'].tolist()
        self.state['target_list'] = self.state['target_list'] + self.state['target'].cpu().numpy().tolist()


        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
              
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})\t'
                      'Acc {acc_current:.4f}\t'
                      'Micro_F1-score {micro_f1_current:.4f}\t'
                      'Macro_F1-score {macro_f1_current:.4f}\t'
                      'Weighted_F1-score {weighted_f1_current:.4f}'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss,
                    acc_current=self.state['acc_batch'],
                    micro_f1_current=self.state['batch_micro_f1'],
                    macro_f1_current=self.state['batch_macro_f1'],
                    weighted_f1_current=self.state['batch_weighted_f1']))
            else:
                print('Val: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})\t'
                      'Acc {acc_current:.4f}\t'
                      'Micro_F1-score {micro_f1_current:.4f}\t'
                      'Macro_F1-score {macro_f1_current:.4f}\t'
                      'Weighted_F1-score {weighted_f1_current:.4f}'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss,
                    acc_current=self.state['acc_batch'],
                    micro_f1_current=self.state['batch_micro_f1'],
                    macro_f1_current=self.state['batch_macro_f1'],
                    weighted_f1_current=self.state['batch_weighted_f1']
                     ))

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):

        input_var = torch.autograd.Variable(self.state['input'])
        target_var = torch.autograd.Variable(self.state['target'])

        if not training:
            input_var.volatile = True
            target_var.volatile = True

        # compute output
        self.state['output'] = model(input_var)
        print('---------------------output.shape---------------------------')
        print((self.state['output']).shape)
        self.state['loss'] = criterion(self.state['output'], target_var)
        ###compute batch_acc
        self.state['pred'] = torch.nn.functional.softmax(self.state['output'], dim=1).argmax(dim=1).cpu().detach().numpy()
        self.state['acc'] = accuracy_score(target_var, self.state['pred'])
        ###compute batch_f1
        self.state['micro_f1'] = f1_score(target_var, self.state['pred'], average='micro')
        self.state['macro_f1'] = f1_score(target_var, self.state['pred'], average='macro')
        self.state['weighted_f1'] = f1_score(target_var, self.state['pred'], average='weighted')

        if training:
            optimizer.zero_grad()
            if  self.state['fp16']:
                with amp.scale_loss(self.state['loss'], optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.state['loss'].backward()
            # self.state['loss'].backward()
            optimizer.step()

    def init_learning(self, model, criterion):

        if self._state('train_transform') is None:
            normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
            self.state['train_transform'] = transforms.Compose([
                MultiScaleCrop(self.state['image_size'], scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        if self._state('val_transform') is None:
            normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
            self.state['val_transform'] = transforms.Compose([
                Warp(self.state['image_size']),
                transforms.ToTensor(),
                normalize,
            ])
        
        if self._state('test_transform') is None:
            normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
            self.state['test_transform'] = transforms.Compose([
                Warp(self.state['image_size']),
                transforms.ToTensor(),
                normalize,
            ])

        ###用来记录最好的val_acc，acc越大越好，进行模型保存
        self.state['best_score'] = 0 
        
        # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
        # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
        # remove the need for this code, but it is still valid.
        if self.state['fp16']:
            try:
                import apex

                apex.amp.register_half_function(torch, "einsum")
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    def learning(self, model, criterion, train_dataset, val_dataset, test_dataset, optimizer=None):

        self.init_learning(model, criterion)

        # define train and val transform
        train_dataset.transform = self.state['train_transform']
        train_dataset.target_transform = self._state('train_target_transform')
        val_dataset.transform = self.state['val_transform']
        val_dataset.target_transform = self._state('val_target_transform')
        test_dataset.transform = self.state['test_transform']
        test_dataset.target_transform = self._state('test_target_transform')


        # data loading code
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.state['batch_size'], shuffle=True,
                                                   num_workers=self.state['workers'])
        print('---------the length of train_loader is {}------------'.format(len(train_loader)))
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=self.state['batch_size'], shuffle=False,
                                                 num_workers=self.state['workers'])
        print('---------the length of val_loader is {}------------'.format(len(val_loader)))


        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=self.state['batch_size'], shuffle=False,
                                                 num_workers=self.state['workers'])
        print('---------the length of test_loader is {}------------'.format(len(test_loader)))

        

        # optionally resume from a checkpoint
        if self._state('resume') is not None:
            if os.path.isfile(self.state['resume']):
                print("=> loading checkpoint '{}'".format(self.state['resume']))
                checkpoint = torch.load(self.state['resume'])
                self.state['start_epoch'] = checkpoint['epoch']
                self.state['best_score'] = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.state['evaluate'], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.state['resume']))


        if self.state['use_gpu']:
            train_loader.pin_memory = True
            val_loader.pin_memory = True
            test_loader.pin_memory = True
            cudnn.benchmark = False
            # model = torch.nn.DataParallel(model, device_ids=self.state['device_ids']).cuda()###平行计算
            device = torch.device('cuda:0' if self.state['use_gpu'] else 'cpu')
            model =model.to(device)
            criterion = criterion.to(device)
            if self.state['fp16']:
                try:
                    from apex import amp
                except ImportError:
                    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

                model, optimizer = amp.initialize(model, optimizer, opt_level=self.state['fp16_opt_level'])

        # TODO define optimizer

        for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
            self.state['epoch'] = epoch
            lr = self.adjust_learning_rate(optimizer)
            print('lr:',lr)

            # train for one epoch
            self.train(train_loader, model, criterion, optimizer, epoch)
            # evaluate on validation set，
            loss, acc, micro_f1, macro_f1, weighted_f1 = self.validate(val_loader, model, criterion)

            # remember best acc and save checkpoint, loss越小越好
            if acc >= self.state['best_score']:
                is_best = True
            else:
                is_best = False
            # is_best = loss < self.state['best_score']
            self.state['best_score'] = max(acc, self.state['best_score'])
            self.save_checkpoint({
                'epoch': epoch + 1,
                'arch': self._state('arch'),
                'state_dict': model.state_dict() if self.state['use_gpu'] else model.state_dict(),
                'best_score': self.state['best_score'],
            }, is_best)
            print(' *** best={best:.4f}'.format(best=self.state['best_score']))

            ###进行testdataset测试
        if self.state['evaluate']:
            target_names = ['angry', 'bored', 'calm', 'fear', 'happy', 'love', 'sad']
            filename_best = 'model_best.pth.tar'
            filename_best = os.path.join(self.state['save_model_path'], filename_best)
            # filename_6896 = 'model_best_0.6907.pth.tar'
            # filename_best = os.path.join(self.state['save_model_path'], filename_6896)
            print("=> loading checkpoint '{}'".format(filename_best))
            checkpoint = torch.load(filename_best)
            self.state['start_epoch'] = checkpoint['epoch']
            self.state['best_score'] = checkpoint['best_score']
            model.load_state_dict(checkpoint['state_dict'])
            print('loading best checkpoint!')
            loss, acc, micro_f1, macro_f1, weighted_f1, id_list, target_list, pred_list = self.test(test_loader, model, criterion)
            print('---------------------Testing----------------------------')
            print('Test: \t '
                  'Loss {loss:.4f}\t'
                  'Acc {acc:.4f}\t'
                  'Micro_f1 {micro_f1:.4f}\t'
                  'Macro_f1 {macro_f1:.4f}\t'
                  'Weighted_f1 {weighted_f1:.4f}'.format(loss=loss, 
                                                     acc=acc,
                                                     micro_f1=micro_f1,
                                                     macro_f1=macro_f1,
                                                     weighted_f1=weighted_f1))
            
            ###针对test的预测结果和label，计算acc，micro_f1, macro_f1, weighted_f1, 看看和前面的计算方式得到的结果一不一致？
            Acc_=accuracy_score(target_list, pred_list)
            Mi_F1 = f1_score(target_list, pred_list, average='micro')
            Ma_F1 = f1_score(target_list, pred_list, average='macro')
            Wg_F1 = f1_score(target_list, pred_list, average='weighted')
            print('---------------------Another Testing----------------------------')
            print('Test_another: \t '
                  'Loss {loss:.4f}\t'
                  'Acc {acc:.4f}\t'
                  'Mi_F1 {micro_f1:.4f}\t'
                  'Ma_F1 {macro_f1:.4f}\t'
                  'Wg_F1 {weighted_f1:.4f}'.format(loss=loss, 
                                                     acc=Acc_,
                                                     micro_f1=Mi_F1,
                                                     macro_f1=Ma_F1,
                                                     weighted_f1=Wg_F1))

            ###result/experiment_result/dataset/model/text_min_count_{}_ngram_{}_winsize_{}_object_t_{}_place_t_{}_img_size_{}_lr_{}_bts_{}.txt
            test_experiment_result_path = 'text_min_count_{}_ngram_{}_winsize_{}_object_t_{}_place_t_{}_img_size_{}_lr_{}_bts_{}.txt'.format(
                                           self.state['text_min_count'],
                                           self.state['ngram'],
                                           self.state['window_size'],
                                           self.state['object_t_value'],
                                           self.state['place_t_value'],
                                           self.state['image_size'], 
                                           self.state['lr'], 
                                           self.state['batch_size'])
            save_experiment_result_path = os.path.join(self.state['save_experiment_result_path'], 
                                                       test_experiment_result_path)
            with open(save_experiment_result_path, 'a', encoding='utf8') as fw:
                fw.write('\n---------------------Testing----------------------------\n')
                fw.write('Test: \t '
                  'Loss {loss:.4f}\t'
                  'Acc {acc:.4f}\t'
                  'Micro_f1 {micro_f1:.4f}\t'
                  'Macro_f1 {macro_f1:.4f}\t'
                  'Weighted_f1 {weighted_f1:.4f}'.format(loss=loss, 
                                                     acc=acc,
                                                     micro_f1=micro_f1,
                                                     macro_f1=macro_f1,
                                                     weighted_f1=weighted_f1))

                fw.write('\n---------------------Another Testing----------------------------')
                fw.write('\n Test_another: \t '
                  'Loss {loss:.4f}\t'
                  'Acc {acc:.4f}\t'
                  'Mi_F1 {micro_f1:.4f}\t'
                  'Ma_F1 {macro_f1:.4f}\t'
                  'Wg_F1 {weighted_f1:.4f}\n'.format(loss=loss, 
                                                     acc=Acc_,
                                                     micro_f1=Mi_F1,
                                                     macro_f1=Ma_F1,
                                                     weighted_f1=Wg_F1))
                fw.write(classification_report(target_list, pred_list, target_names=target_names, digits=4))
                fw.write('\n\n')
            ###result/pred_result/dataset/model/text_min_count_{}_ngram_{}_winsize_{}_object_t_{}_place_t_{}_img_size_{}_lr_{}_bts_{}.txt
            print('--------------------saving pred result--------------------')
            test_pred_result_path = 'text_min_count_{}_ngram_{}_winsize_{}_object_t_{}_place_t_{}_img_size_{}_lr_{}_bts_{}.txt'.format(
                                           self.state['text_min_count'],
                                           self.state['ngram'],
                                           self.state['window_size'],
                                           self.state['object_t_value'],
                                           self.state['place_t_value'],
                                           self.state['image_size'], 
                                           self.state['lr'], 
                                           self.state['batch_size'])
            save_pred_result_path = os.path.join(self.state['save_pred_result_path'], 
                                                       test_pred_result_path)
            with open(save_pred_result_path, 'w') as fp:
                fp.write('ID\tTarget\tPred\n')
                for i in range(len(self.state['id_list'])):
                    fp.write(id_list[i])
                    fp.write('\t')
                    fp.write(str(target_list[i]))
                    fp.write('\t')
                    fp.write(str(pred_list[i]))
                    fp.write('\t')
                    fp.write('\n')
        
        return self.state['best_score']

    def train(self, data_loader, model, criterion, optimizer, epoch):

        # switch to train mode
        model.train()

        self.on_start_epoch(True, model, criterion, data_loader, optimizer)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Training')

        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(True, model, criterion, data_loader, optimizer)

            if self.state['use_gpu']:
                self.state['target'] = self.state['target'].cuda()

            self.on_forward(True, model, criterion, data_loader, optimizer)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(True, model, criterion, data_loader, optimizer)

        loss, acc, micro_f1, macro_f1, weighted_f1, _, _, _ = self.on_end_epoch(True, model, criterion, data_loader, optimizer)

    def validate(self, data_loader, model, criterion):

        # switch to evaluate mode
        model.eval()

        self.on_start_epoch(False, model, criterion, data_loader)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Val')

        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
           
            self.state['target'] = target

            self.on_start_batch(False, model, criterion, data_loader)

            if self.state['use_gpu']:
                self.state['target'] = self.state['target'].cuda()

            self.on_forward(False, model, criterion, data_loader)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(False, model, criterion, data_loader)

        loss, acc, micro_f1, macro_f1, weighted_f1, _, _, _ = self.on_end_epoch(False, model, criterion, data_loader)

        return loss, acc, micro_f1, macro_f1, weighted_f1

    def test(self, data_loader, model, criterion):
        # switch to evaluate mode
        model.eval()
        self.on_start_epoch(False, model, criterion, data_loader)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Test')
        end = time.time()

        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            
            self.state['target'] = target

            self.on_start_batch(False, model, criterion, data_loader)

            if self.state['use_gpu']:
                self.state['target'] = self.state['target'].cuda()

            self.on_forward(False, model, criterion, data_loader)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(False, model, criterion, data_loader)

        loss, acc, micro_f1, macro_f1, weighted_f1, id_list, target_list, pred_list = self.on_end_epoch(False, model, criterion, data_loader)

        return loss, acc, micro_f1, macro_f1, weighted_f1, id_list, target_list, pred_list


    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if self._state('save_model_path') is not None:
            filename_ = filename
            filename = os.path.join(self.state['save_model_path'], filename_)
            if not os.path.exists(self.state['save_model_path']):
                os.makedirs(self.state['save_model_path'])
        print('save model {filename}'.format(filename=filename))
        torch.save(state, filename)
        if is_best:
            # filename_best = 'learning_rate_{}_batch_size_{}_model_best.pth.tar'.format(self.state['lr'], self.state['batch_size'])
            filename_best = 'model_best.pth.tar'
            if self._state('save_model_path') is not None:
                filename_best = os.path.join(self.state['save_model_path'], filename_best)
            shutil.copyfile(filename, filename_best) ###filename为源文件， filename_best为目标文件
            if self._state('save_model_path') is not None:
                if self._state('filename_previous_best') is not None:
                    os.remove(self._state('filename_previous_best'))
                filename_best = os.path.join(self.state['save_model_path'], 'model_best_{score:.4f}.pth.tar'.format(score=state['best_score']))
                shutil.copyfile(filename, filename_best)
                self.state['filename_previous_best'] = filename_best

    def adjust_learning_rate(self, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
        lr_list = []
        decay = 0.2 if sum(self.state['epoch'] == np.array(self.state['epoch_step'])) > 0 else 1.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay
            lr_list.append(param_group['lr'])
        return np.unique(lr_list)


class MultiClassEngine(Engine):
    def __init__(self, state):
        Engine.__init__(self, state)

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        Engine.on_start_epoch(self, training, model, criterion, data_loader, optimizer)

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        loss = self.state['meter_loss'].value()[0]

        self.state['epoch_acc'] = sum(self.state['batch_acc_list'])/len(data_loader)
        acc = self.state['epoch_acc']
        # print('========================self.state[batch_acc_list]========================')
        # print(self.state['batch_acc_list'])
        # print(len(self.state['batch_acc_list']))
        # print('=============================len(data_loader)=====================================')
        # print(len(data_loader))

        ###micro_f1
        self.state['epoch_micro_f1'] = sum(self.state['batch_micro_f1_list'])/len(data_loader) ###计算一个epoch的平均f1,感觉这一句可以直接删掉
        micro_f1 = self.state['epoch_micro_f1']
        ###macro_f1
        self.state['epoch_macro_f1'] = sum(self.state['batch_macro_f1_list'])/len(data_loader) ###计算一个epoch的平均f1
        macro_f1 = self.state['epoch_macro_f1']
        ##weighted_f1
        self.state['epoch_weighted_f1'] = sum(self.state['batch_weighted_f1_list'])/len(data_loader) ###计算一个epoch的平均f1
        weighted_f1 = self.state['epoch_weighted_f1']
        if display:
            if training:
                # print('-----------------------self.state[batch_acc_list]-------------------------')
                # print(len(self.state['batch_acc_list']))
                print('-----------------Epoch: [{0}]\t'
                      'Loss {loss:.4f}\t'
                      'Acc {acc:.4f}\t'
                      'Micro_f1 {micro_f1:.4f}\t'
                      'Macro_f1 {macro_f1:.4f}\t'
                      'Weighted_f1 {weighted_f1:.4f}\n\n'.format(self.state['epoch'],
                                                     loss=loss, 
                                                     acc=acc,
                                                     micro_f1=micro_f1,
                                                     macro_f1=macro_f1,
                                                     weighted_f1=weighted_f1))

            else:
                print('-----------------------self.state[batch_acc_list]-------------------------')
                print(len(self.state['batch_acc_list']))
                print('--------------------Val: \t'
                'Loss {loss:.4f}\t'
                'Acc {acc:.4f}\t'
                'Micro_f1 {micro_f1:.4f}\t'
                'Macro_f1 {macro_f1:.4f}\t'
                'Weighted_f1 {weighted_f1:.4f}\n\n'.format(loss=loss, 
                                                     acc=acc,
                                                     micro_f1=micro_f1,
                                                     macro_f1=macro_f1,
                                                     weighted_f1=weighted_f1))
        return loss, acc, micro_f1, macro_f1, weighted_f1, self.state['id_list'], self.state['target_list'], self.state['pred_list']

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        '''
        dataset:input: (self.id, self.text, self.content, current_len, text_mask, img, image_path, self.object_inp, self.place_inp),
                traget: self.label
        '''
        input = self.state['input']
        self.state['id'] = input[0]
        self.state['text_feature'] = input[2]
        self.state['text_lens'] = input[3]
        self.state['text_mask'] = input[4]
        self.state['object_feature'] = input[5]
        self.state['place_feature'] = input[5]
        self.state['image_name'] = input[6]
        self.state['object_input'] = input[7]
        self.state['place_input'] = input[8]
        target = self.state['target']
        

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        Engine.on_end_batch(self, training, model, criterion, data_loader, optimizer, display=False)


        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]

            # print('----------------------------len(self.state[batch_acc_list]-------------------------------')
            # print(len(self.state['batch_acc_list']))
            # print(self.state['batch_acc_list']) ###1，11，21....
            acc = sum(self.state['batch_acc_list'])/len(self.state['batch_acc_list'])
            batch_acc = self.state['batch_acc'] ###记录的是每十次batch最后一个batch的acc值，因为从0开始，所以是0，10，20....

            micro_f1 = sum(self.state['batch_micro_f1_list'])/len(self.state['batch_micro_f1_list'])
            batch_micro_f1 = self.state['batch_micro_f1']

            macro_f1 = sum(self.state['batch_macro_f1_list'])/len(self.state['batch_macro_f1_list'])
            batch_macro_f1 = self.state['batch_macro_f1']

            weighted_f1 = sum(self.state['batch_weighted_f1_list'])/len(self.state['batch_weighted_f1_list'])
            batch_weighted_f1 = self.state['batch_weighted_f1']

            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})\t'
                      'Acc {acc_current:.4f} ({acc:.4f})\t'
                      'Micro_f1 {micro_f1_current:.4f} ({micro_f1:.4f})\t'
                      'Macro_f1 {macro_f1_current:.4f} ({macro_f1:.4f})\t'
                      'Weighted_f1 {weighted_f1_current:.4f} ({weighted_f1:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss,
                    acc_current=batch_acc, acc=acc,
                    micro_f1_current=batch_micro_f1, micro_f1=micro_f1,
                    macro_f1_current=batch_macro_f1, macro_f1=macro_f1,
                    weighted_f1_current=batch_weighted_f1, weighted_f1=weighted_f1))
            else:
                print('Val: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})\t'
                      'Acc {acc_current:.4f} ({acc:.4f})\t'
                      'Micro_f1 {micro_f1_current:.4f} ({micro_f1:.4f})\t'
                      'Macro_f1 {macro_f1_current:.4f} ({macro_f1:.4f})\t'
                      'Weighted_f1 {weighted_f1_current:.4f} ({weighted_f1:.4f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss,
                    acc_current=batch_acc, acc=acc,
                    micro_f1_current=batch_micro_f1, micro_f1=micro_f1,
                    macro_f1_current=batch_macro_f1, macro_f1=macro_f1,
                    weighted_f1_current=batch_weighted_f1, weighted_f1=weighted_f1))


class GCNMultiClassEngine(MultiClassEngine):
    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
        text_feature_var = torch.autograd.Variable(self.state['text_feature'])
        text_lens = self.state['text_lens']
        text_mask = self.state['text_mask']
        object_feature_var = torch.autograd.Variable(self.state['object_feature']).float()
        place_feature_var = torch.autograd.Variable(self.state['place_feature']).float()
        target_var = torch.autograd.Variable(self.state['target']).float()
        object_inp_var = torch.autograd.Variable(self.state['object_input']).float().detach()  # one hot
        place_inp_var = torch.autograd.Variable(self.state['place_input']).float().detach()  # one hot
    
        device = torch.device('cuda:0' if self.state['use_gpu'] else 'cpu')
        text_feature_var = text_feature_var.to(device)
        text_lens = text_lens.to(device)
        text_mask = text_mask.to(device)
        object_feature_var =  object_feature_var.to(device)
        place_feature_var =  place_feature_var.to(device)
        target_var =  target_var.to(device).long()
        object_inp_var =  object_inp_var.to(device)
        place_inp_var = place_inp_var.to(device)

        if not training:
            with torch.no_grad():
                text_feature_var = text_feature_var
                text_lens = text_lens 
                text_mask = text_mask
                object_feature_var  = object_feature_var 
                place_feature_var =  place_feature_var 
                # print('---------------------type(feature_var) is {}---------------'.format(type(feature_var)))
                target_var = target_var
                object_inp_var = object_inp_var
                place_inp_var = place_inp_var

        '''------------------------------model forward-----------------------------------'''
        self.state['output'] = model(text_feature_var, text_lens, text_mask, object_feature_var, place_feature_var, object_inp_var, place_inp_var)
        self.state['loss'] = criterion(self.state['output'], target_var)
        # self.state['pred'] = torch.nn.functional.softmax(self.state['output'], dim=1).argmax(dim=1).cpu().detach().numpy()
        self.state['output'] = torch.nn.functional.softmax(self.state['output'], dim=1)

        self.state['pred'] = self.state['output'].argmax(dim=1)
        self.state['pred'] = self.state['pred'].cpu().detach().numpy()
        target_var = target_var.cpu().numpy()
        
        self.state['acc'] = accuracy_score(target_var, self.state['pred'])
         ###compute batch_f1
        self.state['micro_f1'] = f1_score(target_var, self.state['pred'], average='micro')
        self.state['macro_f1'] = f1_score(target_var, self.state['pred'], average='macro')
        self.state['weighted_f1'] = f1_score(target_var, self.state['pred'], average='weighted')

        if training:
            optimizer.zero_grad()
            if self.state['fp16']:
                with amp.scale_loss(self.state['loss'], optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.state['loss'].backward()
            if self.state['fp16']:
                    nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_norm=10.0)
            else:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()


    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        
        input = self.state['input']
        self.state['id'] = input[0]
        self.state['text_feature'] = input[2]
        self.state['text_lens'] = input[3]
        self.state['text_mask'] = input[4]
        self.state['object_feature'] = input[5]
        self.state['place_feature'] = input[5]
        self.state['image_name'] = input[6]
        self.state['object_input'] = input[7]
        self.state['place_input'] = input[8]

