# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from niftynet.layer import layer_util
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.deconvolution import DeconvolutionalLayer
from niftynet.layer.downsample import DownSampleLayer
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.layer.crop import CropLayer
from niftynet.utilities.util_common import look_up_operations
import tensorflow as tf


class UNet3D(TrainableLayer):
    """
    reimplementation of 3D U-net
      Çiçek et al., "3D U-Net: Learning dense Volumetric segmentation from
      sparse annotation", MICCAI '16
    """

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='UNet'):
        super(UNet3D, self).__init__(name=name)

        self.n_features = [32, 64, 128, 256, 512]
        self.acti_func = acti_func
        self.num_classes = num_classes

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

        print('using {}'.format(name))

    def layer_op(self, images, is_training=True, layer_id=-1, **unused_kwargs):
        # image_size-4  should be divisible by 8
        assert layer_util.check_spatial_dims(images, lambda x: x % 16 == 4)
        assert layer_util.check_spatial_dims(images, lambda x: x >= 89)
        block_layer = UNetBlock('DOWNSAMPLE',
                                (self.n_features[0], self.n_features[1]),
                                (3, 3), with_downsample_branch=True,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='d0')
        pool_1, conv_1 = block_layer(images, is_training)
        print(block_layer)

        block_layer = UNetBlock('DOWNSAMPLE',
                                (self.n_features[1], self.n_features[2]),
                                (3, 3), with_downsample_branch=True,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='d1')
        pool_2, conv_2 = block_layer(pool_1, is_training)
        print(block_layer)

        block_layer = UNetBlock('DOWNSAMPLE',
                                (self.n_features[2], self.n_features[3]),
                                (3, 3), with_downsample_branch=True,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='d2')
        pool_3, conv_3 = block_layer(pool_2, is_training)
        print(block_layer)

        block_layer = UNetBlock('UPSAMPLE',
                                (self.n_features[3], self.n_features[4]),
                                (3, 3), with_downsample_branch=False,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='d3')
        up_3, _ = block_layer(pool_3, is_training)
        print(block_layer)

        block_layer = UNetBlock('UPSAMPLE',
                                (self.n_features[3], self.n_features[3]),
                                (3, 3), with_downsample_branch=False,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='u2')
        crop_layer = CropLayer(border=4, name='crop-8')
        concat_3 = ElementwiseLayer('CONCAT')(crop_layer(conv_3), up_3)
        up_2, _ = block_layer(concat_3, is_training)
        print(block_layer)

        block_layer = UNetBlock('UPSAMPLE',
                                (self.n_features[2], self.n_features[2]),
                                (3, 3), with_downsample_branch=False,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='u1')
        crop_layer = CropLayer(border=16, name='crop-32')
        concat_2 = ElementwiseLayer('CONCAT')(crop_layer(conv_2), up_2)
        up_1, _ = block_layer(concat_2, is_training)
        print(block_layer)

        block_layer = UNetBlock('NONE',
                                (self.n_features[1],
                                 self.n_features[1],
                                 self.num_classes),
                                (3, 3),
                                with_downsample_branch=True,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='u0')
        crop_layer = CropLayer(border=40, name='crop-80')
        concat_1 = ElementwiseLayer('CONCAT')(crop_layer(conv_1), up_1)
        print(block_layer)

        # for the last layer, upsampling path is not used
        _, output_tensor = block_layer(concat_1, is_training)

        output_conv_op = ConvolutionalLayer(n_output_chns=self.num_classes,
            kernel_size=1,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            acti_func=None,
            name='{}'.format(self.num_classes),
            padding='VALID',
            with_bn=False,
            with_bias=True)
        final_output_tensor = output_conv_op(output_tensor, is_training)
        print(output_conv_op)

        return final_output_tensor

class UNet3D_shallow(TrainableLayer):
    """
    reimplementation of 3D U-net with only two resolution levels
      Çiçek et al., "3D U-Net: Learning dense Volumetric segmentation from
      sparse annotation", MICCAI '16
    """

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='U-Net_3D_shallow'):
        super(UNet3D_shallow, self).__init__(name=name)

        self.n_features = [32, 64, 128]
        self.acti_func = acti_func
        self.num_classes = num_classes

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

        print('using {}'.format(name))

    def layer_op(self, images, is_training=True, layer_id=-1, **unused_kwargs):
        # image_size  should be divisible by 4
        assert layer_util.check_spatial_dims(images, lambda x: x % 4 == 0 )
        assert layer_util.check_spatial_dims(images, lambda x: x >= 21 )
        block_layer = UNetBlock('DOWNSAMPLE',
                                (self.n_features[0], self.n_features[1]),
                                (3, 3), with_downsample_branch=True,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='d0')
        pool_1, conv_1 = block_layer(images, is_training)
        print(block_layer)

        block_layer = UNetBlock('UPSAMPLE',
                                (self.n_features[1], self.n_features[2]),
                                (3, 3), with_downsample_branch=False,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='d1')
        up_1, _ = block_layer(pool_1, is_training)
        print(block_layer)

        block_layer = UNetBlock('NONE',
                                (self.n_features[1],
                                 self.n_features[1],
                                 self.num_classes),
                                (3, 3),
                                with_downsample_branch=True,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='u0')
        crop_layer = CropLayer(border=4, name='crop-8')
        concat_1 = ElementwiseLayer('CONCAT')(crop_layer(conv_1), up_1)
        print(block_layer)

        # for the last layer, upsampling path is not used
        _, output_tensor = block_layer(concat_1, is_training)

        output_conv_op = ConvolutionalLayer(n_output_chns=self.num_classes,
            kernel_size=1,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            acti_func=None,
            name='{}'.format(self.num_classes),
            padding='VALID',
            with_bn=False,
            with_bias=True)
        final_output_tensor = output_conv_op(output_tensor, is_training)
        print(output_conv_op)

        return final_output_tensor

class UNet3D_anisotropic(TrainableLayer):
    """
    re-implementation of the 3D U-net with anisotpic filters
      Çiçek et al., "3D U-Net: Learning dense Volumetric segmentation from
      sparse annotation", MICCAI '16
    """

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='U-Net_3D_anisotropic'):
        super(UNet3D_anisotropic, self).__init__(name=name)

        self.n_features = [32, 64, 128, 256, 512]
        self.acti_func = acti_func
        self.num_classes = num_classes

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

        print('using {}'.format(name))

    def layer_op(self, images, is_training=True, layer_id=-1, **unused_kwargs):
        # image_size-4  should be divisible by 8
        #assert layer_util.check_spatial_dims(images, lambda x: x % 16 == 4)
        #assert layer_util.check_spatial_dims(images, lambda x: x >= 89)
        block_layer = UNetBlock('DOWNSAMPLE_ANISOTROPIC',
                                (self.n_features[0], self.n_features[1]),
                                ([3,3,1], [3,3,1]), with_downsample_branch=True,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='d0')
        pool_1, conv_1 = block_layer(images, is_training)
        print(block_layer)

        block_layer = UNetBlock('DOWNSAMPLE_ANISOTROPIC',
                                (self.n_features[1], self.n_features[2]),
                                ([3,3,1], [3,3,1]), with_downsample_branch=True,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='d1')
        pool_2, conv_2 = block_layer(pool_1, is_training)
        print(block_layer)

        block_layer = UNetBlock('DOWNSAMPLE',
                                (self.n_features[2], self.n_features[3]),
                                (3, 3), with_downsample_branch=True,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='d2')
        pool_3, conv_3 = block_layer(pool_2, is_training)
        print(block_layer)

        block_layer = UNetBlock('UPSAMPLE',
                                (self.n_features[3], self.n_features[4]),
                                (3, 3), with_downsample_branch=False,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='d3')
        up_3, _ = block_layer(pool_3, is_training)
        print(block_layer)

        block_layer = UNetBlock('UPSAMPLE_ANISOTROPIC',
                                (self.n_features[3], self.n_features[3]),
                                (3, 3), with_downsample_branch=False,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='u2')
        crop_layer = CropLayer(border=4, name='crop-8')
        concat_3 = ElementwiseLayer('CONCAT')(crop_layer(conv_3), up_3)
        up_2, _ = block_layer(concat_3, is_training)
        print(block_layer)

        block_layer = UNetBlock('UPSAMPLE_ANISOTROPIC',
                                (self.n_features[2], self.n_features[2]),
                                ([3,3,1], [3,3,1]), with_downsample_branch=False,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='u1')
        crop_layer = CropLayer(border=[16,16,8], name='crop-32x32x16')
        concat_2 = ElementwiseLayer('CONCAT')(crop_layer(conv_2), up_2)
        up_1, _ = block_layer(concat_2, is_training)
        print(block_layer)

        block_layer = UNetBlock('NONE',
                                (self.n_features[1],
                                 self.n_features[1],
                                 self.num_classes),
                                ([3,3,1], [3,3,1]),
                                with_downsample_branch=True,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='u0')
        crop_layer = CropLayer(border=[40,40,8], name='crop-80x8x16')
        concat_1 = ElementwiseLayer('CONCAT')(crop_layer(conv_1), up_1)
        print(block_layer)

        # for the last layer, upsampling path is not used
        _, output_tensor = block_layer(concat_1, is_training)

        output_conv_op = ConvolutionalLayer(n_output_chns=self.num_classes,
            kernel_size=1,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            acti_func=None,
            name='{}'.format(self.num_classes),
            padding='VALID',
            with_bn=False,
            with_bias=True)
        final_output_tensor = output_conv_op(output_tensor, is_training)
        print(output_conv_op)

        return final_output_tensor

class UNet3D_anisotropic_3x(TrainableLayer):
    """
    re-implementation of the 3D U-net with anisotpic filters in three layers.
      Çiçek et al., "3D U-Net: Learning dense Volumetric segmentation from
      sparse annotation", MICCAI '16
    """

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='U-Net_3D_anisotropic_3x'):
        super(UNet3D_anisotropic_3x, self).__init__(name=name)

        self.n_features = [32, 64, 128, 256, 512]
        self.acti_func = acti_func
        self.num_classes = num_classes

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

        print('using {}'.format(name))

    def layer_op(self, images, is_training=True, layer_id=-1, **unused_kwargs):
        # image_size-4  should be divisible by 8
        #assert layer_util.check_spatial_dims(images, lambda x: x % 16 == 4)
        #assert layer_util.check_spatial_dims(images, lambda x: x >= 89)
        block_layer = UNetBlock('DOWNSAMPLE_ANISOTROPIC',
                                (self.n_features[0], self.n_features[1]),
                                ([3,3,1], [3,3,1]), with_downsample_branch=True,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='d0')
        pool_1, conv_1 = block_layer(images, is_training)
        print(block_layer)

        block_layer = UNetBlock('DOWNSAMPLE_ANISOTROPIC',
                                (self.n_features[1], self.n_features[2]),
                                ([3,3,1], [3,3,1]), with_downsample_branch=True,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='d1')
        pool_2, conv_2 = block_layer(pool_1, is_training)
        print(block_layer)

        block_layer = UNetBlock('DOWNSAMPLE_ANISOTROPIC',
                                (self.n_features[2], self.n_features[3]),
                                ([3,3,1], [3,3,1]), with_downsample_branch=True,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='d2')
        pool_3, conv_3 = block_layer(pool_2, is_training)
        print(block_layer)

        block_layer = UNetBlock('UPSAMPLE_ANISOTROPIC',
                                (self.n_features[3], self.n_features[4]),
                                ([3,3,3], [3,3,3]), with_downsample_branch=False,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='d3')
        up_3, _ = block_layer(pool_3, is_training)
        print(block_layer)

        block_layer = UNetBlock('UPSAMPLE_ANISOTROPIC',
                                (self.n_features[3], self.n_features[3]),
                                ([3,3,1], [3,3,1]), with_downsample_branch=False,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='u2')
        crop_layer = CropLayer(border=[4,4,2], name='crop-8x8x0')
        concat_3 = ElementwiseLayer('CONCAT')(crop_layer(conv_3), up_3)
        up_2, _ = block_layer(concat_3, is_training)
        print(block_layer)

        block_layer = UNetBlock('UPSAMPLE_ANISOTROPIC',
                                (self.n_features[2], self.n_features[2]),
                                ([3,3,1], [3,3,1]), with_downsample_branch=False,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='u1')
        crop_layer = CropLayer(border=[16,16,2], name='crop-32x32x0')
        concat_2 = ElementwiseLayer('CONCAT')(crop_layer(conv_2), up_2)
        up_1, _ = block_layer(concat_2, is_training)
        print(block_layer)

        block_layer = UNetBlock('NONE',
                                (self.n_features[1],
                                 self.n_features[1],
                                 self.num_classes),
                                ([3,3,1], [3,3,1]),
                                with_downsample_branch=True,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='u0')
        crop_layer = CropLayer(border=[40,40,2], name='crop-80x80x0')
        concat_1 = ElementwiseLayer('CONCAT')(crop_layer(conv_1), up_1)
        print(block_layer)

        # for the last layer, upsampling path is not used
        _, output_tensor = block_layer(concat_1, is_training)

        output_conv_op = ConvolutionalLayer(n_output_chns=self.num_classes,
            kernel_size=1,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            acti_func=None,
            name='{}'.format(self.num_classes),
            padding='VALID',
            with_bn=False,
            with_bias=True)
        final_output_tensor = output_conv_op(output_tensor, is_training)
        print(output_conv_op)

        return final_output_tensor

SUPPORTED_OP = set(['DOWNSAMPLE', 'UPSAMPLE', 'NONE', 'DOWNSAMPLE_ANISOTROPIC', 'UPSAMPLE_ANISOTROPIC'])



class UNetBlock(TrainableLayer):
    def __init__(self,
                 func,
                 n_chns,
                 kernels,
                 w_initializer=None,
                 w_regularizer=None,
                 with_downsample_branch=False,
                 acti_func='relu',
                 name='UNet_block'):

        super(UNetBlock, self).__init__(name=name)

        self.func = look_up_operations(func.upper(), SUPPORTED_OP)

        self.kernels = kernels
        self.n_chns = n_chns
        self.with_downsample_branch = with_downsample_branch
        self.acti_func = acti_func

        self.initializers = {'w': w_initializer}
        self.regularizers = {'w': w_regularizer}

    def layer_op(self, input_tensor, is_training):
        output_tensor = input_tensor
        for (kernel_size, n_features) in zip(self.kernels, self.n_chns):
            conv_op = ConvolutionalLayer(n_output_chns=n_features,
                                         kernel_size=kernel_size,
                                         w_initializer=self.initializers['w'],
                                         w_regularizer=self.regularizers['w'],
                                         acti_func=self.acti_func,
                                         name='{}'.format(n_features),
                                         padding='VALID',
                                         with_bn=False,
                                         with_bias=True)
            output_tensor = conv_op(output_tensor, is_training)

        if self.with_downsample_branch:
            branch_output = output_tensor
        else:
            branch_output = None

        if self.func == 'DOWNSAMPLE':
            downsample_op = DownSampleLayer('MAX',
                                            kernel_size=2,
                                            stride=2,
                                            name='down_2x_isotropic')
            output_tensor = downsample_op(output_tensor)
        if self.func == 'DOWNSAMPLE_ANISOTROPIC':
            downsample_op = DownSampleLayer('MAX',
                                            kernel_size=[2,2,1],
                                            stride=[2,2,1],
                                            name='down_2x2x1')
            output_tensor = downsample_op(output_tensor)
        elif self.func == 'UPSAMPLE':
            upsample_op = DeconvolutionalLayer(n_output_chns=self.n_chns[-1],
                                               kernel_size=2,
                                               stride=2,
                                               name='up_2x_isotropic',
                                               with_bn=False,
                                               with_bias=True)
            output_tensor = upsample_op(output_tensor, is_training)
        elif self.func == 'UPSAMPLE_ANISOTROPIC':
            upsample_op = DeconvolutionalLayer(n_output_chns=self.n_chns[-1],
                                               kernel_size=[2,2,1],
                                               stride=[2,2,1],
                                               name='up_2x2x1',
                                               with_bn=False,
                                               with_bias=True)
            output_tensor = upsample_op(output_tensor, is_training)
        elif self.func == 'NONE':
            pass  # do nothing
        return output_tensor, branch_output
