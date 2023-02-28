import torch
import torch.nn as nn
import math

DISCRIMINATOR_REGISTRY = {}

def register_discriminator(name):
    def register_discriminator_cls(cls):
        if name in DISCRIMINATOR_REGISTRY:
            raise ValueError('Cannot register duplicate module ({})'.format(name))
        DISCRIMINATOR_REGISTRY[name] = cls
        return cls
    return register_discriminator_cls


def get_discriminator(name):
    if name not in DISCRIMINATOR_REGISTRY:
        raise ValueError('Cannot find module {}'.format(name))
    return DISCRIMINATOR_REGISTRY[name]


@register_discriminator('abstract_discriminator')
class Discriminator(nn.Module):
    """
    backbone network
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, **kwargs):
        raise NotImplementedError
    
    @staticmethod
    def add_args(parser):
        pass

@register_discriminator('patch_discriminator')
class NLayerDiscriminator(Discriminator):
    """Defines a PatchGAN discriminator"""

    def __init__(self, args):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__(args)
        input_nc, ndf, n_layers = args.input_nc, args.ndf, args.n_layers
        if args.gan_norm_layer == 'batch':
            norm_layer = nn.BatchNorm2d
            use_bias = False
        elif args.gan_norm_layer == 'instance':
            norm_layer = nn.InstanceNorm2d
            use_bias = True

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--input-nc', type=int, default=3,
                            help='image input channel')
        parser.add_argument('--ndf', type=int, default=64,
                            help='the number of filters in the last conv layer')
        parser.add_argument('--n-layers', type=int, default=3,
                            help='the number of conv layers in the discriminator')
        parser.add_argument('--patch-size', type=int, default=32,
                            help='image patch size that inputs to the discriminator')
        parser.add_argument('--gan-norm-layer', type=str, default='instance',
                            help='batch normalization or instance normalization')

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
            

@register_discriminator('pixel_discriminator')
class PixelDiscriminator(Discriminator):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, args):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super().__init__(args)
        input_nc, ndf = args.input_nc, args.ndf
        if args.gan_norm_layer == 'batch':
            norm_layer = nn.BatchNorm2d
            use_bias = False
        elif args.gan_norm_layer == 'instance':
            norm_layer = nn.InstanceNorm2d
            use_bias = True

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--input-nc', type=int, default=3,
                            help='image input channel')
        parser.add_argument('--ndf', type=int, default=64,
                            help='the number of filters in the last conv layer')
        parser.add_argument('--patch-size', type=int, default=32,
                            help='image patch size that inputs to the discriminator')
        parser.add_argument('--gan-norm-layer', type=str, default='batch',
                            help='batch normalization or instance normalization')

    def forward(self, input):
        """Standard forward."""
        return self.net(input)        