
import logging
import math
import os
import random
import sys
import time

import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR

from fairnr.tasks.neural_rendering import SingleObjRenderingTask
from fairnr.models.nsvf import NSVFModel, my_base_architecture
from fairnr.criterions.rendering_loss import SRNLossCriterion

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('RGBDNeRF.train')


class PolynomialDecayLRSchedule(object):
    """Decay the LR on a fixed schedule."""

    def __init__(self, total_num_update, lr, optimizer, warmup_updates=0, force_anneal=None, end_learning_rate=0.0, power=1.0):
        super().__init__()

        assert total_num_update > 0

        self.lr = lr
        if warmup_updates > 0:
            self.warmup_factor = 1.0 / warmup_updates
        else:
            self.warmup_factor = 1
        self.warmup_updates = warmup_updates
        self.force_anneal = force_anneal
        self.end_learning_rate = end_learning_rate
        self.total_num_update = total_num_update
        self.power = power
        self.optimizer = optimizer
        # self.set_lr(self.warmup_factor * self.lr) # set lr when we define optimizer

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def get_next_lr(self, epoch):
        lrs = self.lr
        if self.force_anneal is None or epoch < self.force_anneal:
            # use fixed LR schedule
            next_lr = lrs[min(epoch, len(lrs) - 1)]
        else:
            # annneal based on lr_shrink
            next_lr = self.get_lr()
        return next_lr

    def step_begin_epoch(self, epoch):
        """Update the learning rate at the beginning of the given epoch."""
        self.lr = self.get_next_lr(epoch)
        self.set_lr(self.warmup_factor * self.lr)
        return self.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if self.warmup_updates > 0 and num_updates <= self.warmup_updates:
            self.warmup_factor = num_updates / float(self.warmup_updates)
            lr = self.warmup_factor * self.lr
        elif num_updates >= self.total_num_update:
            lr = self.end_learning_rate
        else:
            warmup = self.warmup_updates
            lr_range = self.lr - self.end_learning_rate
            pct_remaining = 1 - (num_updates - warmup) / (self.total_num_update - warmup)
            lr = lr_range * pct_remaining ** (self.power) + self.end_learning_rate
        self.set_lr(lr)
        return self.get_lr()

    def step(self, num_updates):
        return self.step_update(num_updates)


def my_train():
    parser = argparse.ArgumentParser()
    SingleObjRenderingTask.add_args(parser)
    NSVFModel.add_args(parser)
    SRNLossCriterion.add_args(parser)
    args = parser.parse_args()
    my_base_architecture(args)
    print(args)

    task = SingleObjRenderingTask(args)
    gpu_id = args.gpu_id.split(',')[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    torch.cuda.set_device(int(gpu_id))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    if args.load_pretrain and os.path.exists(os.path.join(args.save_dir, 'checkpoint_mesh_pretrain.pt')):
        ckpt = torch.load(os.path.join(args.save_dir, 'checkpoint_mesh_pretrain.pt'))
        logger.info('checkpoint load!')
        start_epoch = ckpt['epoch'] + 1
        model_state = ckpt['model']
        if 'lr' in ckpt:
            lr = ckpt['lr']
    elif os.path.exists(os.path.join(args.save_dir, 'checkpoint_last.pt')):
        ckpt = torch.load(os.path.join(args.save_dir, 'checkpoint_last.pt'))
        logger.info('checkpoint load!')
        start_epoch = ckpt['epoch'] + 1
        model_state = ckpt['model']
        if 'lr' in ckpt:
            lr = ckpt['lr']
    else:
        start_epoch = 0
        model_state = None
        lr = args.lr
    
    # load dataset
    task.load_dataset('train')
    task.load_dataset('valid')
    # itr = task.get_batch_iterator(task.datasets['train'])
    train_loader = DataLoader(
        task.datasets['train'], collate_fn=task.datasets['train'].collater, batch_size=args.batch_size, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        task.datasets['valid'], collate_fn=task.datasets['valid'].dataset.collater, batch_size=args.batch_size, num_workers=args.num_workers
    )

    if args.mesh_data is not None and start_epoch < args.mesh_pretrain_num:
        mesh_pretrain = True
        task.load_mesh_dataset('train')
        task.load_mesh_dataset('valid')
        mesh_train_loader = DataLoader(
            task.mesh_datasets['train'], collate_fn=task.mesh_datasets['train'].collater, batch_size=args.batch_size, num_workers=args.num_workers
        )
        mesh_test_loader = DataLoader(
            task.mesh_datasets['valid'], collate_fn=task.mesh_datasets['valid'].dataset.collater, batch_size=args.batch_size, num_workers=args.num_workers
        )
    else:
        mesh_pretrain = False

    # Build model and criterion
    model = task.build_model(args).cuda()
    criterion = task.build_criterion(args)
    # logger.info(model)
    logger.info('model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    logger.info('num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))
    if model_state is not None:
        model.load_state_dict(model_state)

    model.encoder.max_hits = torch.scalar_tensor(args.max_hits)

    if len(args.tensorboard_logdir) > 0:
        from torch.utils.tensorboard import SummaryWriter
        train_writer = SummaryWriter(args.tensorboard_logdir + '/train')
        valid_writer = SummaryWriter(args.tensorboard_logdir + '/valid')
        image_writer = SummaryWriter(args.tensorboard_logdir + '/images')
    else:
        train_writer, valid_writer, image_writer = None, None, None

    if args.dis:
        if args.optimizer == 'adam':
            optimizer = optim.Adam([{'params':model.reader.parameters()}, {'params':model.encoder.parameters()}, {'params':model.field.parameters()}, {'params':model.raymarcher.parameters()}], lr=lr, betas=eval(args.adam_betas))
            dis_optimizer = optim.Adam(model.discriminator.parameters(), lr=lr, betas=eval(args.adam_betas))
        elif args.optimizer == 'rmsprop':
            optimizer = optim.RMSprop([{'params':model.reader.parameters()}, {'params':model.encoder.parameters()}, {'params':model.field.parameters()}, {'params':model.raymarcher.parameters()}], lr=lr, alpha=0.99, eps=1e-8)
            dis_optimizer = optim.RMSprop(model.discriminator.parameters(), lr=lr, alpha=0.99, eps=1e-8)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD([{'params':model.reader.parameters()}, {'params':model.encoder.parameters()}, {'params':model.field.parameters()}, {'params':model.raymarcher.parameters()}], lr=lr, momentum=0.)
            dis_optimizer = optim.SGD(model.discriminator.parameters(), lr=lr, momentum=0.)
    else:
        if args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif args.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-8)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.)
    if args.lr_scheduler == 'polynomial_decay':
        scheduler = PolynomialDecayLRSchedule(args.total_num_update, args.lr, optimizer)
    else:
        scheduler = StepLR(optimizer, int(args.lr_step), gamma=0.99)
    

    for epoch in range(start_epoch, args.total_num_update):
        if mesh_pretrain and epoch <= args.mesh_pretrain_num:
            loss, sample_size, logging_output, _ = task.train_step(mesh_train_loader, model, criterion, optimizer, scheduler, epoch)
            logger.info('epoch: {} | stage: {} | loss: {:.4f} | color: {:.4f} | alpha: {:.4f} | voxs_log: {:.4f} | stps_log: {:.4f} | tvox_log: {:.4f} | asf_log: {:.4f} | ash_log: {:.4f} | lr: {}'.format(
                epoch, 'mesh_pretrain', logging_output['loss'], logging_output['color_loss'], logging_output['alpha_loss'], logging_output['voxs_log'], 
                logging_output['stps_log'], logging_output['tvox_log'], logging_output['asf_log'], logging_output['ash_log'], logging_output['lr']))
        else:
            loss, sample_size, logging_output, _ = task.train_step(train_loader, model, criterion, optimizer, scheduler, epoch)
            if args.dis:
                for param_group in dis_optimizer.param_groups:
                    param_group["lr"] = optimizer.param_groups[0]["lr"]
                _, _, _, loss_D = task.dis_train_step(train_loader, model, criterion, dis_optimizer)
                logger.info('epoch: {} | stage: {} | loss: {:.4f} | color: {:.4f} | alpha: {:.4f} | gen: {:.4f} | dis: {:.4f} | voxs_log: {:.4f} | stps_log: {:.4f} | tvox_log: {:.4f} | asf_log: {:.4f} | ash_log: {:.4f} | lr: {}'.format(
                    epoch, 'real_train', logging_output['loss'], logging_output['color_loss'], logging_output['alpha_loss'], logging_output['gen_loss'], loss_D.data.item(), logging_output['voxs_log'], 
                    logging_output['stps_log'], logging_output['tvox_log'], logging_output['asf_log'], logging_output['ash_log'], logging_output['lr']))
            else:
                logger.info('epoch: {} | stage: {} | loss: {:.4f} | color: {:.4f} | alpha: {:.4f} | voxs_log: {:.4f} | stps_log: {:.4f} | tvox_log: {:.4f} | asf_log: {:.4f} | ash_log: {:.4f} | lr: {}'.format(
                    epoch, 'real_train', logging_output['loss'], logging_output['color_loss'], logging_output['alpha_loss'], logging_output['voxs_log'], 
                    logging_output['stps_log'], logging_output['tvox_log'], logging_output['asf_log'], logging_output['ash_log'], logging_output['lr']))
        
        if train_writer is not None:
            for key in logging_output.keys():
                if 'loss' in key:
                    train_writer.add_scalar(key, logging_output[key], epoch)

        if epoch == args.mesh_pretrain_num:
            torch.save(
                {'model': model.state_dict(), 'args': args, 'epoch':epoch, 'lr':logging_output['lr']},
                f'{args.save_dir}/checkpoint_mesh_pretrain.pt',
            )

        if epoch > 0 and args.save_interval_updates > 0 and epoch % args.save_interval_updates == 0:
            last_epoch = epoch - args.keep_last_epochs * args.save_interval_updates
            if last_epoch > 0 and os.path.exists(os.path.join(args.save_dir,'checkpoint_'+str(last_epoch)+'.pt')):
                os.remove(os.path.join(args.save_dir,'checkpoint_'+str(last_epoch)+'.pt'))
            torch.save(
                {'model': model.state_dict(), 'args': args, 'epoch':epoch, 'lr':logging_output['lr']},
                f'{args.save_dir}/checkpoint_{str(epoch)}.pt',
            )
            torch.save(
                {'model': model.state_dict(), 'args': args, 'epoch':epoch, 'lr':logging_output['lr']},
                f'{args.save_dir}/checkpoint_last.pt',
            )
            logger.info('Save checkpoint!')

            if mesh_pretrain and epoch <= args.mesh_pretrain_num:
                for step, sample in enumerate(mesh_test_loader):
                    sample = {key: sample[key].cuda() for key in sample.keys() if isinstance(sample[key], torch.Tensor)}
                    valid_loss, valid_sample_size, valid_logging_output, valid_loss_D = task.valid_step(sample, model, criterion, image_writer)
                    logger.info('epoch: {} | stage: {} | valid: {}/{} | loss: {:.4f} | color: {:.4f} | alpha: {:.4f} | voxs_log: {:.4f} | stps_log: {:.4f} | tvox_log: {:.4f} | asf_log: {:.4f} | ash_log: {:.4f}'.format(
                        epoch, 'mesh_pretrain', step, len(mesh_test_loader), valid_logging_output['loss'], valid_logging_output['color_loss'], valid_logging_output['alpha_loss'], valid_logging_output['voxs_log'], 
                        valid_logging_output['stps_log'], valid_logging_output['tvox_log'], valid_logging_output['asf_log'], valid_logging_output['ash_log']))
            else:
                for step, sample in enumerate(test_loader):
                    sample = {key: sample[key].cuda() for key in sample.keys() if isinstance(sample[key], torch.Tensor)}
                    valid_loss, valid_sample_size, valid_logging_output, valid_loss_D = task.valid_step(sample, model, criterion, image_writer)
                    logger.info('epoch: {} | stage: {} | valid: {}/{} | loss: {:.4f} | color: {:.4f} | alpha: {:.4f} | voxs_log: {:.4f} | stps_log: {:.4f} | tvox_log: {:.4f} | asf_log: {:.4f} | ash_log: {:.4f}'.format(
                        epoch, 'real_train', step, len(test_loader), valid_logging_output['loss'], valid_logging_output['color_loss'], valid_logging_output['alpha_loss'], valid_logging_output['voxs_log'], 
                        valid_logging_output['stps_log'], valid_logging_output['tvox_log'], valid_logging_output['asf_log'], valid_logging_output['ash_log']))
            if valid_writer is not None:
                for key in valid_logging_output.keys():
                    if 'loss' in key:
                        valid_writer.add_scalar(key, valid_logging_output[key], epoch)

    logger.info('done training!')


if __name__ == '__main__':
    my_train()
