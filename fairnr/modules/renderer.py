
import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairnr.modules.linear import FCLayer
from fairnr.data.geometry import ray
import os
import numpy as np
from fairnr.data.geometry import offset_points, trilinear_interp

MAX_DEPTH = 10000.0
RENDERER_REGISTRY = {}

def register_renderer(name):
    def register_renderer_cls(cls):
        if name in RENDERER_REGISTRY:
            raise ValueError('Cannot register duplicate module ({})'.format(name))
        RENDERER_REGISTRY[name] = cls
        return cls
    return register_renderer_cls


def get_renderer(name):
    if name not in RENDERER_REGISTRY:
        raise ValueError('Cannot find module {}'.format(name))
    return RENDERER_REGISTRY[name]


@register_renderer('abstract_renderer')
class Renderer(nn.Module):
    """
    Abstract class for ray marching
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, **kwargs):
        raise NotImplementedError
    
    @staticmethod
    def add_args(parser):
        pass


@register_renderer('volume_rendering')
class VolumeRenderer(Renderer):

    def __init__(self, args):
        super().__init__(args) 
        self.chunk_size = 1024 * getattr(args, "chunk_size", 64)
        self.valid_chunk_size = 1024 * getattr(args, "valid_chunk_size", self.chunk_size // 1024)
        self.ray_chunk_size = 1024 * getattr(args, "ray_chunk_size", 64)
        self.discrete_reg = getattr(args, "discrete_regularization", False)
        self.raymarching_tolerance = getattr(args, "raymarching_tolerance", 0.0)

    @staticmethod
    def add_args(parser):
        # ray-marching parameters
        parser.add_argument('--discrete-regularization', action='store_true',
                            help='if set, a zero mean unit variance gaussian will be added to encougrage discreteness')
        
        # additional arguments
        parser.add_argument('--chunk-size', type=int, metavar='D', 
                            help='set chunks to go through the network (~K forward passes). trade time for memory. ')
        parser.add_argument('--valid-chunk-size', type=int, metavar='D', 
                            help='chunk size used when no training. In default the same as chunk-size.')
        parser.add_argument('--ray-chunk-size', type=int, metavar='D', 
                            help='ray chunk size used when no training. In default the same as chunk-size.')
        parser.add_argument('--raymarching-tolerance', type=float, default=0)
    
    def forward_once(
        self, input_fn, field_fn, ray_start, ray_dir, samples, encoder_states,
        early_stop=None, output_types=['sigma', 'texture']
        ):
        """
        chunks: set > 1 if out-of-memory. it can save some memory by time.
        """
        sampled_depth = samples['sampled_point_depth']
        sampled_dists = samples['sampled_point_distance']
        sampled_idx = samples['sampled_point_voxel_idx'].long()
        
        # only compute when the ray hits
        sample_mask = sampled_idx.ne(-1)
        if early_stop is not None:
            sample_mask = sample_mask & (~early_stop.unsqueeze(-1))
        if sample_mask.sum() == 0:  # miss everything skip
            return None, 0

        sampled_xyz = ray(ray_start.unsqueeze(1), ray_dir.unsqueeze(1), sampled_depth.unsqueeze(2))
        sampled_dir = ray_dir.unsqueeze(1).expand(*sampled_depth.size(), ray_dir.size()[-1])
        samples['sampled_point_xyz'] = sampled_xyz
        samples['sampled_point_ray_direction'] = sampled_dir
    
        # apply mask    
        samples = {name: s[sample_mask] for name, s in samples.items()}
       
        # get encoder features as inputs
        field_inputs = input_fn(samples, encoder_states)

        if 'voxel_color' in encoder_states:
            voxel_color = encoder_states['voxel_color'].float()
            sampled_idx = samples['sampled_point_voxel_idx'].long()
            voxel_color_feat = F.embedding(sampled_idx, voxel_color)
        else:
            voxel_color_feat = None

        def masked_scatter(mask, x):
            B, K = mask.size()
            if x.dim() == 1:
                return x.new_zeros(B, K).masked_scatter(mask, x)
            return x.new_zeros(B, K, x.size(-1)).masked_scatter(
                mask.unsqueeze(-1).expand(B, K, x.size(-1)), x)
            
        # forward implicit fields
        field_outputs = field_fn(field_inputs, outputs=output_types, color_feat=voxel_color_feat)
        outputs = {'sample_mask': sample_mask}
        
        # post processing
        if 'sigma' in field_outputs:
            sigma, sampled_dists= field_outputs['sigma'], samples['sampled_point_distance']
            noise = 0 if not self.discrete_reg and (not self.training) else torch.zeros_like(sigma).normal_()  
            free_energy = torch.relu(noise + sigma) * sampled_dists    
            free_energy = free_energy * 7.0    # magic operation
            # free_energy = (F.elu(sigma - 3, alpha=1) + 1) * sampled_dists
            # (optional) free_energy = (F.elu(sigma - 3, alpha=1) + 1) * dists
            outputs['free_energy'] = masked_scatter(sample_mask, free_energy)
        if 'texture' in field_outputs:
            if self.args.res:
                field_outputs['texture'] = field_outputs['texture'] + voxel_color_feat
            outputs['texture'] = masked_scatter(sample_mask, field_outputs['texture'])
        if 'normal' in field_outputs:
            outputs['normal'] = masked_scatter(sample_mask, field_outputs['normal'])
        return outputs, sample_mask.sum()

    def forward_chunk(
        self, input_fn, field_fn, ray_start, ray_dir, samples, encoder_states,
        gt_depths=None, output_types=['sigma', 'texture'], global_weights=None,
        ):
        sampled_depth = samples['sampled_point_depth']
        sampled_idx = samples['sampled_point_voxel_idx'].long()

        tolerance = self.raymarching_tolerance
        chunk_size = self.chunk_size if self.training else self.valid_chunk_size
        early_stop = None
        if tolerance > 0:
            tolerance = -math.log(tolerance)
            
        hits = sampled_idx.ne(-1).long()
        outputs = defaultdict(lambda: [])
        size_so_far, start_step = 0, 0
        accumulated_free_energy = 0
        accumulated_evaluations = 0
        for i in range(hits.size(1) + 1):
            if ((i == hits.size(1)) or (size_so_far + hits[:, i].sum() > chunk_size)) and (i > start_step):
                _outputs, _evals = self.forward_once(
                        input_fn, field_fn, 
                        ray_start, ray_dir, 
                        {name: s[:, start_step: i] 
                            for name, s in samples.items()},
                        encoder_states, 
                        early_stop=early_stop,
                        output_types=output_types)
                if _outputs is not None:
                    accumulated_evaluations += _evals

                    if 'free_energy' in _outputs:
                        accumulated_free_energy += _outputs['free_energy'].sum(1)
                        if tolerance > 0:
                            early_stop = accumulated_free_energy > tolerance
                            hits[early_stop] *= 0
                    
                    for key in _outputs:
                        outputs[key] += [_outputs[key]]
                else:
                    for key in outputs:
                        outputs[key] += [outputs[key][-1].new_zeros(
                            outputs[key][-1].size(0),
                            sampled_depth[:, start_step: i].size(1),
                            *outputs[key][-1].size()[2:] 
                        )]
                start_step, size_so_far = i, 0
            
            if (i < hits.size(1)):
                size_so_far += hits[:, i].sum()

        outputs = {key: torch.cat(outputs[key], 1) for key in outputs}
        results = {}
        
        if 'free_energy' in outputs:
            free_energy = outputs['free_energy']
            shifted_free_energy = torch.cat([free_energy.new_zeros(sampled_depth.size(0), 1), free_energy[:, :-1]], dim=-1)  # shift one step
            a = 1 - torch.exp(-free_energy.float())                             # probability of it is not empty here
            b = torch.exp(-torch.cumsum(shifted_free_energy.float(), dim=-1))   # probability of everything is empty up to now
            probs = (a * b).type_as(free_energy)                                # probability of the ray hits something here
        else:
            probs = outputs['sample_mask'].type_as(sampled_depth) / sampled_depth.size(-1)  # assuming a uniform distribution

        if global_weights is not None:
            probs = probs * global_weights
            
        depth = (sampled_depth * probs).sum(-1)
        missed = 1 - probs.sum(-1)
        results.update({'probs': probs, 'depths': depth, 'missed': missed, 'ae': accumulated_evaluations})

        if 'texture' in outputs:
            results['colors'] = (outputs['texture'] * probs.unsqueeze(-1)).sum(-2)
        if 'normal' in outputs:
            results['normal'] = (outputs['normal'] * probs.unsqueeze(-1)).sum(-2)
        return results

    def forward(self, input_fn, field_fn, ray_start, ray_dir, samples, encoder_states, *args, **kwargs):
        chunk_size = self.chunk_size if self.training else self.valid_chunk_size
        ray_chunk_size = self.ray_chunk_size
        if ray_start.size(0) <= chunk_size:
            results = self.forward_chunk(input_fn, field_fn, ray_start, ray_dir, samples, encoder_states, *args, **kwargs)
            return results

        # the number of rays is larger than maximum forward passes. pre-chuncking..
        results = [
            self.forward_chunk(input_fn, field_fn, 
                ray_start[i: i+chunk_size], ray_dir[i: i+chunk_size],
                {name: s[i: i+chunk_size] for name, s in samples.items()}, encoder_states, *args, **kwargs)
            for i in range(0, ray_start.size(0), chunk_size)
        ]
        results = {name: torch.cat([r[name] for r in results], 0) 
                    if results[0][name].dim() > 0 else sum([r[name] for r in results])
                for name in results[0]}

        if getattr(input_fn, "track_max_probs", False):
            input_fn.track_voxel_probs(samples['sampled_point_voxel_idx'].long(), results['probs'])
        return results
        
