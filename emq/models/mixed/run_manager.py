import json
import math
import os
import sys
import time
from datetime import timedelta

import apex
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from apex.parallel import DistributedDataParallel as DDP
from torchprofile import profile_macs

from .utils import (AverageMeter, accuracy, count_parameters,
                    cross_entropy_with_label_smoothing, get_unpruned_weights)

# from apex import amp, optimizers


class RunConfig:

    def __init__(self, n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
                 dataset, train_batch_size, test_batch_size, opt_type,
                 opt_param, weight_decay, label_smoothing, no_decay_keys,
                 model_init, init_div_groups, validation_frequency,
                 print_frequency, local_rank, world_size, sync_bn, warm_epoch):
        self.n_epochs = n_epochs
        self.init_lr = init_lr
        self.lr_schedule_type = lr_schedule_type
        self.lr_schedule_param = lr_schedule_param
        self.warm_epoch = warm_epoch

        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        self.opt_type = opt_type
        self.opt_param = opt_param
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.no_decay_keys = no_decay_keys

        self.model_init = model_init
        self.init_div_groups = init_div_groups
        self.validation_frequency = validation_frequency
        self.print_frequency = print_frequency

        self._data_provider = None
        self._train_iter, self._valid_iter, self._test_iter = None, None, None
        self.local_rank = local_rank
        self.world_size = world_size
        self.sync_bn = sync_bn

    @property
    def config(self):
        config = {}
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config

    def copy(self):
        return RunConfig(**self.config)

    """ learning rate """

    def _calc_learning_rate(self, epoch, batch=0, nBatch=None, warm_epoch=5):
        if self.lr_schedule_type == 'cosine':
            T_total = self.n_epochs * nBatch
            T_cur = epoch * nBatch + batch
            T_warm = warm_epoch * nBatch
            if T_cur < T_warm:
                lr = T_cur / T_warm * self.init_lr
            else:
                lr = 0.5 * self.init_lr * \
                    (1 + math.cos(math.pi * (T_cur - T_warm) / (T_total - T_warm)))
        else:
            raise ValueError('do not support: %s' % self.lr_schedule_type)
        return lr

    def adjust_learning_rate(self, optimizer, epoch, batch=0, nBatch=None):
        """ adjust learning of a given optimizer and return the new learning rate """
        new_lr = self._calc_learning_rate(epoch, batch, nBatch,
                                          self.warm_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    """ data provider """

    @property
    def data_config(self):
        raise NotImplementedError

    @property
    def data_provider(self):
        if self._data_provider is None:
            if self.dataset == 'imagenet':
                from data_providers.imagenet import ImagenetDataProvider
                self._data_provider = ImagenetDataProvider(**self.data_config)
            else:
                raise ValueError('do not support: %s' % self.dataset)
        return self._data_provider

    @data_provider.setter
    def data_provider(self, val):
        self._data_provider = val

    @property
    def train_loader(self):
        return self.data_provider.train

    @property
    def valid_loader(self):
        return self.data_provider.valid

    @property
    def test_loader(self):
        return self.data_provider.test

    """ optimizer """

    def build_optimizer(self, net_params):
        if self.opt_type == 'sgd':
            opt_param = {} if self.opt_param is None else self.opt_param
            momentum, nesterov = opt_param.get('momentum', 0.9), opt_param.get(
                'nesterov', True)
            if self.no_decay_keys:
                optimizer = torch.optim.SGD([
                    {
                        'params': net_params[0],
                        'weight_decay': self.weight_decay
                    },
                    {
                        'params': net_params[1],
                        'weight_decay': 0
                    },
                ],
                                            lr=self.init_lr,
                                            momentum=momentum,
                                            nesterov=nesterov)
            else:
                optimizer = torch.optim.SGD(
                    net_params,
                    self.init_lr,
                    momentum=momentum,
                    nesterov=nesterov,
                    weight_decay=self.weight_decay)
        else:
            raise NotImplementedError
        return optimizer


class RunManager:

    def __init__(self, path, net, run_config: RunConfig, out_log=True):
        self.path = path
        self.net = net
        self.run_config = run_config
        self.out_log = out_log

        self._logs_path, self._save_path = None, None
        self.best_acc = 0
        self.start_epoch = 0
        gpu = self.run_config.local_rank
        torch.cuda.set_device(gpu)

        # initialize model (default)
        self.net.init_model(run_config.model_init, run_config.init_div_groups)

        # net info
        self.net = self.net.cuda()
        if run_config.local_rank == 0:
            self.print_net_info()

        if self.run_config.sync_bn:
            self.net = apex.parallel.convert_syncbn_model(self.net)
        print('local_rank: %d' % self.run_config.local_rank)

        self.run_config.init_lr = self.run_config.init_lr * float(
            self.run_config.train_batch_size *
            self.run_config.world_size) / 256.
        self.criterion = nn.CrossEntropyLoss()
        if self.run_config.no_decay_keys:
            keys = self.run_config.no_decay_keys.split('#')
            self.optimizer = self.run_config.build_optimizer([
                # parameters with weight decay
                self.net.get_parameters(keys, mode='exclude'),
                # parameters without weight decay
                self.net.get_parameters(keys, mode='include'),
            ])
        else:
            self.optimizer = self.run_config.build_optimizer(
                self.net.weight_parameters())
        # self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level='O1')
        self.net = DDP(self.net, delay_allreduce=True)
        cudnn.benchmark = True

    """ save path and log path """

    @property
    def save_path(self):
        if self._save_path is None:
            save_path = os.path.join(self.path, 'checkpoint')
            os.makedirs(save_path, exist_ok=True)
            self._save_path = save_path
        return self._save_path

    @property
    def logs_path(self):
        if self._logs_path is None:
            logs_path = os.path.join(self.path, 'logs')
            os.makedirs(logs_path, exist_ok=True)
            self._logs_path = logs_path
        return self._logs_path

    """ net info """

    def reset_model(self, model, model_origin=None):
        self.net = model
        self.net.init_model(self.run_config.model_init,
                            self.run_config.init_div_groups)
        if model_origin is not None:
            if self.run_config.local_rank == 0:
                print('-' * 30 + ' start pruning ' + '-' * 30)
            get_unpruned_weights(self.net, model_origin)
            if self.run_config.local_rank == 0:
                print('-' * 30 + ' end pruning ' + '-' * 30)
        # net info
        self.net = self.net.cuda()
        if self.run_config.local_rank == 0:
            self.print_net_info()

        if self.run_config.sync_bn:
            self.net = apex.parallel.convert_syncbn_model(self.net)
        print('local_rank: %d' % self.run_config.local_rank)

        self.criterion = nn.CrossEntropyLoss()
        if self.run_config.no_decay_keys:
            keys = self.run_config.no_decay_keys.split('#')
            self.optimizer = self.run_config.build_optimizer([
                # parameters with weight decay
                self.net.get_parameters(keys, mode='exclude'),
                # parameters without weight decay
                self.net.get_parameters(keys, mode='include'),
            ])
        else:
            self.optimizer = self.run_config.build_optimizer(
                self.net.weight_parameters())
        # model, self.optimizer = amp.initialize(model, self.optimizer,
        #                                        opt_level='O2',
        #                                        keep_batchnorm_fp32=True,
        #                                        loss_scale=1.0
        #                                        )
        self.net = DDP(self.net, delay_allreduce=True)
        cudnn.benchmark = True
        # if model_origin!=None:
        #     if self.run_config.local_rank==0:
        #         print('-'*30+' start training bn '+'-'*30)
        #     self.train_bn(1)
        #     if self.run_config.local_rank==0:
        #         print('-'*30+' end training bn '+'-'*30)

    # noinspection PyUnresolvedReferences
    def net_flops(self):
        data_shape = [1] + list(self.run_config.data_provider.data_shape)

        net = self.net
        input_var = torch.zeros(data_shape).cuda()
        with torch.no_grad():
            flops = profile_macs(net, input_var)
        return flops

    def print_net_info(self):
        # parameters
        total_params = count_parameters(self.net)
        if self.out_log:
            print('Total training params: %.2fM' % (total_params / 1e6))
        net_info = {
            'param': '%.2fM' % (total_params / 1e6),
        }

        # flops
        flops = self.net_flops()
        if self.out_log:
            print('Total FLOPs: %.1fM' % (flops / 1e6))
        net_info['flops'] = '%.1fM' % (flops / 1e6)

        # config
        if self.out_log:
            print('Net config: ' + str(self.net.config))
        net_info['config'] = str(self.net.config)

        with open('%s/net_info.txt' % self.logs_path, 'w') as fout:
            fout.write(json.dumps(net_info, indent=4) + '\n')

    """ save and load models """

    def save_model(self, checkpoint=None, is_best=False, model_name=None):
        if checkpoint is None:
            checkpoint = {'state_dict': self.net.module.state_dict()}

        if model_name is None:
            model_name = 'checkpoint.pth.tar'

        # add `dataset` info to the checkpoint
        checkpoint['dataset'] = self.run_config.dataset
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        model_path = os.path.join(self.save_path, model_name)
        with open(latest_fname, 'w') as fout:
            fout.write(model_path + '\n')
        torch.save(checkpoint, model_path)

        if is_best:
            best_path = os.path.join(self.save_path, 'model_best.pth.tar')
            torch.save({'state_dict': checkpoint['state_dict']}, best_path)

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, 'r') as fin:
                model_fname = fin.readline()
                if model_fname[-1] == '\n':
                    model_fname = model_fname[:-1]
        # noinspection PyBroadException
        try:
            if model_fname is None or not os.path.exists(model_fname):
                model_fname = '%s/checkpoint.pth.tar' % self.save_path
                with open(latest_fname, 'w') as fout:
                    fout.write(model_fname + '\n')
            if self.out_log:
                print("=> loading checkpoint '{}'".format(model_fname))

            if torch.cuda.is_available():
                checkpoint = torch.load(model_fname)
            else:
                checkpoint = torch.load(model_fname, map_location='cpu')

            self.net.module.load_state_dict(checkpoint['state_dict'])
            # set new manual seed
            new_manual_seed = int(time.time())
            torch.manual_seed(new_manual_seed)
            torch.cuda.manual_seed_all(new_manual_seed)
            np.random.seed(new_manual_seed)

            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
            if 'best_acc' in checkpoint:
                self.best_acc = checkpoint['best_acc']
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            if self.out_log:
                print("=> loaded checkpoint '{}'".format(model_fname))
        except Exception:
            if self.out_log:
                print('fail to load checkpoint from %s' % self.save_path)

    def save_config(self, print_info=True):
        """ dump run_config and net_config to the model_folder """
        os.makedirs(self.path, exist_ok=True)
        net_save_path = os.path.join(self.path, 'net.config')
        json.dump(self.net.module.config, open(net_save_path, 'w'), indent=4)
        if print_info:
            print('Network configs dump to %s' % net_save_path)

        run_save_path = os.path.join(self.path, 'run.config')
        json.dump(self.run_config.config, open(run_save_path, 'w'), indent=4)
        if print_info:
            print('Run configs dump to %s' % run_save_path)

    """ train and test """

    def write_log(self, log_str, prefix, should_print=True):
        """ prefix: valid, train, test """
        if prefix in ['valid', 'test']:
            with open(os.path.join(self.logs_path, 'valid_console.txt'),
                      'a') as fout:
                fout.write(log_str + '\n')
                fout.flush()
        if prefix in ['valid', 'test', 'train']:
            with open(os.path.join(self.logs_path, 'train_console.txt'),
                      'a') as fout:
                if prefix in ['valid', 'test']:
                    fout.write('=' * 10)
                fout.write(log_str + '\n')
                fout.flush()
        if prefix in ['prune']:
            with open(os.path.join(self.logs_path, 'prune_console.txt'),
                      'a') as fout:
                if prefix in ['valid', 'test']:
                    fout.write('=' * 10)
                fout.write(log_str + '\n')
                fout.flush()
        if should_print:
            print(log_str)

    def validate(self,
                 is_test=True,
                 net=None,
                 use_train_mode=False,
                 return_top5=False):
        if is_test:
            data_loader = self.run_config.test_loader
        else:
            data_loader = self.run_config.valid_loader

        if net is None:
            net = self.net

        if use_train_mode:
            net.train()
        else:
            net.eval()
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end = time.time()
        # noinspection PyUnresolvedReferences
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                images, labels = data[0].cuda(non_blocking=True), data[1].cuda(
                    non_blocking=True)
                # images, labels = data[0].cuda(), data[1].cuda()
                # compute output
                output = net(images)
                loss = self.criterion(output, labels.long())
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                reduced_loss = self.reduce_tensor(loss.data)
                acc1 = self.reduce_tensor(acc1)
                acc5 = self.reduce_tensor(acc5)
                losses.update(reduced_loss, images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.run_config.print_frequency == 0 or i + 1 == len(
                        data_loader):
                    if is_test:
                        prefix = 'Test'
                    else:
                        prefix = 'Valid'
                    test_log = prefix + ': [{0}/{1}]\t' \
                                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                                        'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})'. \
                        format(i, len(data_loader) - 1,
                               batch_time=batch_time, loss=losses, top1=top1)
                    if return_top5:
                        test_log += '\tTop-5 acc {top5.val:.3f} ({top5.avg:.3f})'.format(
                            top5=top5)
                    print(test_log)
        self.run_config.valid_loader.reset()
        self.run_config.test_loader.reset()
        if return_top5:
            return losses.avg, top1.avg, top5.avg
        else:
            return losses.avg, top1.avg

    def train_bn(self, epochs=1):
        if self.run_config.local_rank == 0:
            print('training bn')
        for m in self.net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean = torch.zeros_like(m.running_mean)
                m.running_var = torch.ones_like(m.running_var)
        self.net.train()
        for i in range(epochs):
            for _, data in enumerate(self.run_config.train_loader):
                images, labels = data[0].cuda(non_blocking=True), data[1].cuda(
                    non_blocking=True)
                output = self.net(images)
                del output, images, labels
        if self.run_config.local_rank == 0:
            print('training bn finished')

    def train_one_epoch(self, adjust_lr_func, train_log_func, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        self.net.train()

        end = time.time()
        for i, data in enumerate(self.run_config.train_loader):
            data_time.update(time.time() - end)
            new_lr = adjust_lr_func(i)
            images, labels = data[0].cuda(non_blocking=True), data[1].cuda(
                non_blocking=True)
            # compute output
            output = self.net(images)
            if self.run_config.label_smoothing > 0:
                loss = cross_entropy_with_label_smoothing(
                    output, labels, self.run_config.label_smoothing)
            else:
                loss = self.criterion(output, labels)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            reduced_loss = self.reduce_tensor(loss.data)
            acc1 = self.reduce_tensor(acc1)
            acc5 = self.reduce_tensor(acc5)
            losses.update(reduced_loss, images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            self.net.zero_grad()  # or self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            torch.cuda.synchronize()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i % self.run_config.print_frequency == 0
                    or i + 1 == len(self.run_config.train_loader)
                ) and self.run_config.local_rank == 0:
                batch_log = train_log_func(i, batch_time, data_time, losses,
                                           top1, top5, new_lr)
                self.write_log(batch_log, 'train')
        return top1, top5

    def train(self, print_top5=False):

        def train_log_func(epoch_, i, batch_time, data_time, losses, top1,
                           top5, lr):
            batch_log = 'Train [{0}][{1}/{2}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                        'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                        'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})'. \
                format(epoch_ + 1, i, len(self.run_config.train_loader) - 1,
                       batch_time=batch_time, data_time=data_time, losses=losses, top1=top1)
            if print_top5:
                batch_log += '\tTop-5 acc {top5.val:.3f} ({top5.avg:.3f})'.format(
                    top5=top5)
            batch_log += '\tlr {lr:.5f}'.format(lr=lr)
            return batch_log

        for epoch in range(self.start_epoch, self.run_config.n_epochs):
            if self.run_config.local_rank == 0:
                print('\n', '-' * 30, 'Train epoch: %d' % (epoch + 1),
                      '-' * 30, '\n')

            end = time.time()
            train_top1, train_top5 = self.train_one_epoch(
                lambda i: self.run_config.adjust_learning_rate(
                    self.optimizer, epoch, i, len(self.run_config.train_loader)
                ), lambda i, batch_time, data_time, losses, top1, top5, new_lr:
                train_log_func(epoch, i, batch_time, data_time, losses, top1,
                               top5, new_lr), epoch)
            time_per_epoch = time.time() - end
            seconds_left = int(
                (self.run_config.n_epochs - epoch - 1) * time_per_epoch)
            if self.run_config.local_rank == 0:
                print('Time per epoch: %s, Est. complete in: %s' %
                      (str(timedelta(seconds=time_per_epoch)),
                       str(timedelta(seconds=seconds_left))))

            if (epoch + 1) % self.run_config.validation_frequency == 0:
                val_loss, val_acc, val_acc5 = self.validate(
                    is_test=False, return_top5=True)
                is_best = val_acc > self.best_acc
                self.best_acc = max(self.best_acc, val_acc)
                val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f} ({4:.3f})'. \
                    format(epoch + 1, self.run_config.n_epochs,
                           val_loss, val_acc, self.best_acc)
                if print_top5:
                    val_log += '\ttop-5 acc {0:.3f}\tTrain top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}'. \
                        format(val_acc5, top1=train_top1, top5=train_top5)
                else:
                    val_log += '\tTrain top-1 {top1.avg:.3f}'.format(
                        top1=train_top1)
                if self.run_config.local_rank == 0:
                    self.write_log(val_log, 'valid')
            else:
                is_best = False
            if self.run_config.local_rank == 0:
                self.save_model(
                    {
                        'epoch': epoch,
                        'best_acc': self.best_acc,
                        'optimizer': self.optimizer.state_dict(),
                        'state_dict': self.net.state_dict(),
                    },
                    is_best=is_best)
            self.run_config.train_loader.reset()
            self.run_config.valid_loader.reset()
            self.run_config.test_loader.reset()

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.run_config.world_size
        return rt
