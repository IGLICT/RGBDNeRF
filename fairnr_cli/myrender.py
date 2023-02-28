#!/usr/bin/env python3 -u

import logging
import math
import os
import sys
import time
import torch
import imageio
import numpy as np

from fairnr import options

import argparse
from torch.utils.data import DataLoader
from fairnr.tasks.neural_rendering import SingleObjRenderingTask
from fairnr.models.nsvf import NSVFModel, my_base_architecture
from fairnr.criterions.rendering_loss import SRNLossCriterion

def my_generation():
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        stream=sys.stdout,
    )
    logger = logging.getLogger('RGBDNeRF.render')

    parser = argparse.ArgumentParser()
    SingleObjRenderingTask.add_args(parser)
    NSVFModel.add_args(parser)
    SRNLossCriterion.add_args(parser)
    options.add_rendering_args(parser)
    args = parser.parse_args()

    assert args.path is not None, '--path required for generation!'
    ckpt = torch.load(args.path)
    if 'args' in ckpt:
        load_args = ckpt['args']

    # args = {key: load_args[key] for key in args.keys() if key in load_args.keys() else args[key]}
    for key in vars(args).keys():
        if key in vars(load_args).keys():
            setattr(args, key, getattr(load_args, key))

    if args.model_overrides is not None:
        arg_overrides = eval(args.model_overrides)
        for key in arg_overrides:
            setattr(args, key, arg_overrides[key])
    
    my_base_architecture(args)
    print(args)
    task = SingleObjRenderingTask(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    # load dataset
    task.load_dataset('test')
    gen_loader = DataLoader(
        task.datasets['test'], collate_fn=task.datasets['test'].collater, batch_size=1, num_workers=0
    )

    # Build mocel and generator
    model = task.build_model(args).cuda()
    generator = task.build_generator(args)
    model.load_state_dict(ckpt['model'])
    if generator.test_poses is not None:
        frames = generator.test_poses.shape[0]
    else:
        frames = args.render_num_frames

    output_files, step= [], 0
    for i, sample in enumerate(gen_loader):
        sample = {key: sample[key].cuda() for key in sample.keys() if isinstance(sample[key], torch.Tensor)}
        step, _output_files = task.inference_step(generator, [model], [sample, step, frames])
        output_files += _output_files
        print(step)

    generator.save_images(output_files, combine_output=args.render_combine_output)

if __name__ == '__main__':
    my_generation()
