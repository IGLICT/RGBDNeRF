
import math

import torch.nn.functional as F
import torch
from torch import Tensor

import fairnr.criterions.utils as utils

def item(tensor):
    # tpu-comment: making this a no-op for xla devices.
    if torch.is_tensor(tensor) and tensor.device.type == 'xla':
        return tensor.detach()
    if hasattr(tensor, "item"):
        return tensor.item()
    if hasattr(tensor, "__getitem__"):
        return tensor[0]
    return tensor

class RenderingCriterion(object):

    def __init__(self, args):
        super().__init__()
        self.args = args

    @classmethod
    def build_criterion(cls, args):
        """Construct a criterion from command-line args."""
        return cls(args)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        pass

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample)
        loss, loss_output, loss_D = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = 1
        
        logging_output = {
            'loss': loss.data.item() if reduce else loss.data,
            'nsentences': sample['alpha'].size(0),
            'ntokens':  sample['alpha'].size(1),
            'npixels': sample['alpha'].size(2),
            'sample_size': sample_size,
        }
        for w in loss_output:
            logging_output[w] = loss_output[w]
        
        return loss, sample_size, logging_output, loss_D

    def compute_loss(self, model, net_output, sample, reduce=True):
        raise NotImplementedError

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


class SRNLossCriterion(RenderingCriterion):

    def __init__(self, args):
        super().__init__(args)
        # HACK: to avoid warnings in c10d
        self.dummy_loss = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=True)  
        if args.vgg_weight > 0:
            from fairnr.criterions.perceptual_loss import VGGPerceptualLoss
            self.vgg = VGGPerceptualLoss(resize=False)

        if args.eval_lpips: # not use???
            from lpips import LPIPS
            self.lpips = LPIPS(net='alex')

        if args.gan_weight > 0:
            from fairnr.criterions.gan_loss import GANLoss
            self.criterionGAN = GANLoss(args.gan_mode)
            
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--L1', action='store_true',
                            help='if enabled, use L1 instead of L2 for RGB loss')
        parser.add_argument('--color-weight', type=float, default=256.0)
        parser.add_argument('--depth-weight', type=float, default=0.0)
        parser.add_argument('--depth-weight-decay', type=str, default=None,
                            help="""if set, use tuple to set (final_ratio, steps).
                                    For instance, (0, 30000)    
                                """)
        parser.add_argument('--alpha-weight', type=float, default=0.0)
        parser.add_argument('--vgg-weight', type=float, default=0.0)
        parser.add_argument('--vgg-level', type=int, choices=[1,2,3,4], default=2)
        parser.add_argument('--eval-lpips', action='store_true',
                            help="evaluate LPIPS scores in validation")
        parser.add_argument('--no-background-loss', action='store_true')
        parser.add_argument('--gan-weight', type=float, default=0.0)
        parser.add_argument('--gan-mode', type=str, default='lsgan', 
            help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')

    def compute_loss(self, model, net_output, sample, reduce=True):
        losses, other_logs = {}, {}
        
        # prepare data before computing loss
        sampled_uv = net_output['sampled_uv']  # S, V, 2, N, P, P (patch-size)
        S, V, _, N, P1, P2 = sampled_uv.size()
        # H, W, h, w = sample['size'][0, 0].long().cpu().tolist()
        H, W, h, w = sample['size'][0, 0].cpu().tolist()
        L = N * P1 * P2
        flatten_uv = sampled_uv.view(S, V, 2, L)
        flatten_index = (flatten_uv[:,:,0] // h + flatten_uv[:,:,1] // w * W).long()

        assert 'colors' in sample and sample['colors'] is not None, "ground-truth colors not provided"
        target_colors = sample['colors']
        masks = (sample['alpha'] > 0) if self.args.no_background_loss else None
        if L < target_colors.size(2):
            target_colors = target_colors.gather(2, flatten_index.unsqueeze(-1).repeat(1,1,1,3))
            masks = masks.gather(2, flatten_uv) if masks is not None else None
    
        if 'other_logs' in net_output:
            other_logs.update(net_output['other_logs'])

        # computing loss
        if self.args.color_weight > 0:
            color_loss = utils.rgb_loss(
                net_output['colors'], target_colors, 
                masks, self.args.L1)
            losses['color_loss'] = (color_loss, self.args.color_weight)
        
        if self.args.alpha_weight > 0:
            _alpha = net_output['missed'].reshape(-1)
            alpha_loss = torch.log1p(
                1. / 0.11 * _alpha.float() * (1 - _alpha.float())
            ).mean().type_as(_alpha)
            losses['alpha_loss'] = (alpha_loss, self.args.alpha_weight)

        if self.args.depth_weight > 0:
            if sample['depths'] is not None:
                target_depths = sample['depths']
                target_depths = target_depths.gather(2, flatten_index)
                depth_mask = masks & (target_depths > 0) if masks is not None else None
                depth_loss = utils.depth_loss(net_output['depths'], target_depths, depth_mask)
                
            else:
                # no depth map is provided, depth loss only applied on background based on masks
                max_depth_target = self.args.max_depth * torch.ones_like(net_output['depths'])
                if sample['mask'] is not None:        
                    depth_loss = utils.depth_loss(net_output['depths'], max_depth_target, (1 - sample['mask']).bool())
                else:
                    depth_loss = utils.depth_loss(net_output['depths'], max_depth_target, ~masks)
            
            depth_weight = self.args.depth_weight
            if self.args.depth_weight_decay is not None:
                final_factor, final_steps = eval(self.args.depth_weight_decay)
                depth_weight *= max(0, 1 - (1 - final_factor) * self.task._num_updates / final_steps)
                other_logs['depth_weight'] = depth_weight

            losses['depth_loss'] = (depth_loss, depth_weight)

        
        if self.args.vgg_weight > 0:
            assert P1 * P2 > 1, "we have to use a patch-based sampling for VGG loss"
            target_colors = target_colors.reshape(-1, P1, P2, 3).permute(0, 3, 1, 2) * .5 + .5
            output_colors = net_output['colors'].reshape(-1, P1, P2, 3).permute(0, 3, 1, 2) * .5 + .5
            vgg_loss = self.vgg(output_colors, target_colors)
            losses['vgg_loss'] = (vgg_loss, self.args.vgg_weight)

        if self.args.gan_weight > 0 and 'pred_fake' in net_output:
            pred_fake, pred_real = net_output['pred_fake'], net_output['pred_real']
            loss_G = self.criterionGAN(pred_fake, True)
            loss_D_fake = self.criterionGAN(pred_fake.detach(), False)
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            losses['gen_loss'] = (loss_G, self.args.gan_weight)
        else:
            loss_D = None

        loss = sum(losses[key][0] * losses[key][1] for key in losses)
       
        logging_outputs = {key: item(losses[key][0]) for key in losses}
        logging_outputs.update(other_logs)
        return loss, logging_outputs, loss_D
