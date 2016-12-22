__author__ = 'Simon Kohl'

from lasagne.layers import InputLayer, ConcatLayer, TransposedConv2DLayer, NonlinearityLayer, DimshuffleLayer, ReshapeLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.init import HeNormal

from collections import OrderedDict

def build_net(input_dim=572, no_channels=3, seg_entities=2):
    """Implementation of 'U-Net: Convolutional Networks for Biomedical Image Segmentation',
       https://arxiv.org/pdf/1505.04597.pdf

    :param input_dim: x and y dimensions of 3D input
    :param no_channels: z dimension of 3D input
    :param seg_entities: number of classes to segment, i.e. number of categories per pixel for the softmax function
    """

    nonlin = rectify
    pad = 'valid'

    net = OrderedDict()

    net['input'] = InputLayer((None, no_channels, input_dim, input_dim))

    net['encode/conv1_1'] = ConvLayer(net['input'], 64, 3, pad=pad, nonlinearity=nonlin, W=HeNormal(gain="relu"))
    net['encode/conv1_2'] = ConvLayer(net['encode/conv1_1'], 64, 3, pad=pad, nonlinearity=nonlin, W=HeNormal(gain="relu"))
    net['encode/pool1'] = PoolLayer(net['encode/conv1_2'], 2)

    net['encode/conv2_1'] = ConvLayer(net['encode/pool1'], 128, 3, pad=pad, nonlinearity=nonlin, W=HeNormal(gain="relu"))
    net['encode/conv2_2'] = ConvLayer(net['encode/conv2_1'], 128, 3, pad=pad, nonlinearity=nonlin, W=HeNormal(gain="relu"))
    net['encode/pool2'] = PoolLayer(net['encode/conv2_2'], 2)

    net['encode/conv3_1'] = ConvLayer(net['encode/pool2'], 256, 3, pad=pad, nonlinearity=nonlin, W=HeNormal(gain="relu"))
    net['encode/conv3_2'] = ConvLayer(net['encode/conv3_1'], 256, 3, pad=pad, nonlinearity=nonlin, W=HeNormal(gain="relu"))
    net['encode/pool3'] = PoolLayer(net['encode/conv3_2'], 2)

    net['encode/conv4_1'] = ConvLayer(net['encode/pool3'], 512, 3, pad=pad, nonlinearity=nonlin, W=HeNormal(gain="relu"))
    net['encode/conv4_2'] = ConvLayer(net['encode/conv4_1'], 512, 3, pad=pad, nonlinearity=nonlin, W=HeNormal(gain="relu"))
    net['encode/pool4'] = PoolLayer(net['encode/conv4_2'], 2)

    net['encode/conv5_1'] = ConvLayer(net['encode/pool4'], 1024, 3, pad=pad, nonlinearity=nonlin, W=HeNormal(gain="relu"))
    net['encode/conv5_2'] = ConvLayer(net['encode/conv5_1'], 1024, 3, pad=pad, nonlinearity=nonlin, W=HeNormal(gain="relu"))

    net['decode/up_conv1'] = TransposedConv2DLayer(net['encode/conv5_2'], 512, 2, stride=2, crop='valid', nonlinearity=None)
    net['decode/concat_c4_u1'] = ConcatLayer([net['encode/conv4_2'], net['decode/up_conv1']], axis=1, cropping=(None, None, 'center', 'center'))
    net['decode/conv1_1'] = ConvLayer(net['decode/concat_c4_u1'], 512, 3, pad=pad, nonlinearity=nonlin, W=HeNormal(gain="relu"))
    net['decode/conv1_2'] = ConvLayer(net['decode/conv1_1'], 512, 3, pad=pad, nonlinearity=nonlin, W=HeNormal(gain="relu"))

    net['decode/up_conv2'] = TransposedConv2DLayer(net['decode/conv1_2'], 256, 2, stride=2, crop='valid', nonlinearity=None)
    net['decode/concat_c3_u2'] = ConcatLayer([net['encode/conv3_2'], net['decode/up_conv2']], axis=1, cropping=(None, None, 'center', 'center'))
    net['decode/conv2_1'] = ConvLayer(net['decode/concat_c3_u2'], 256, 3, pad=pad, nonlinearity=nonlin, W=HeNormal(gain="relu"))
    net['decode/conv2_2'] = ConvLayer(net['decode/conv2_1'], 256, 3, pad=pad, nonlinearity=nonlin, W=HeNormal(gain="relu"))

    net['decode/up_conv3'] = TransposedConv2DLayer(net['decode/conv2_2'], 128, 2, stride=2, crop='valid', nonlinearity=None)
    net['decode/concat_c2_u3'] = ConcatLayer([net['encode/conv2_2'], net['decode/up_conv3']], axis=1, cropping=(None, None, 'center', 'center'))
    net['decode/conv3_1'] = ConvLayer(net['decode/concat_c2_u3'], 128, 3, pad=pad, nonlinearity=nonlin, W=HeNormal(gain="relu"))
    net['decode/conv3_2'] = ConvLayer(net['decode/conv3_1'], 128, 3, pad=pad, nonlinearity=nonlin, W=HeNormal(gain="relu"))

    net['decode/up_conv4'] = TransposedConv2DLayer(net['decode/conv3_2'], 64, 2, stride=2, crop='valid', nonlinearity=None)
    net['decode/concat_c1_u4'] = ConcatLayer([net['encode/conv1_2'], net['decode/up_conv4']], axis=1, cropping=(None, None, 'center', 'center'))
    net['decode/conv4_1'] = ConvLayer(net['decode/concat_c1_u4'], 128, 3, pad=pad, nonlinearity=nonlin, W=HeNormal(gain="relu"))
    net['decode/conv4_2'] = ConvLayer(net['decode/conv4_1'], 128, 3, pad=pad, nonlinearity=nonlin, W=HeNormal(gain="relu"))

    net['seg_map'] = ConvLayer(net['decode/conv4_2'], seg_entities, 1, pad=pad, nonlinearity=None, W=HeNormal(gain="relu"))

    net['seg_map/dimshuffle'] = DimshuffleLayer(net['seg_map'], (1, 0, 2, 3))
    net['seg_map/reshape'] = ReshapeLayer(net['seg_map/dimshuffle'], (seg_entities, -1))
    net['seg_map/flattened'] = DimshuffleLayer(net['seg_map/reshape'], (1, 0))
    net['out'] = NonlinearityLayer(net['seg_map/flattened'], nonlinearity=softmax)

    return net







