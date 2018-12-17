# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/10/10 11:09
#project: Face recognize
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  papers:
####################################################
import keras 
import keras.layers as KL
from keras.regularizers import l2
from keras.models import Model
from keras.utils import plot_model
import numpy as np
#for depthconv
from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
try:
    from keras.engine import InputSpec
except ImportError:
    from keras.engine.topology import InputSpec

class DepthwiseConv2D(KL.Conv2D):
    """Depthwise separable 2D convolution.

    Depthwise Separable convolutions consists in performing
    just the first step in a depthwise spatial convolution
    (which acts on each input channel separately).
    The `depth_multiplier` argument controls how many
    output channels are generated per input channel in the depthwise step.

    # Arguments
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `'valid'` or `'same'` (case-insensitive).
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be 'channels_last'.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. 'linear' activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        depthwise_initializer: Initializer for the depthwise kernel matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        depthwise_regularizer: Regularizer function applied to
            the depthwise kernel matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its 'activation').
            (see [regularizer](../regularizers.md)).
        depthwise_constraint: Constraint function applied to
            the depthwise kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        4D tensor with shape:
        `[batch, channels, rows, cols]` if data_format='channels_first'
        or 4D tensor with shape:
        `[batch, rows, cols, channels]` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `[batch, filters, new_rows, new_cols]` if data_format='channels_first'
        or 4D tensor with shape:
        `[batch, new_rows, new_cols, filters]` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 depth_multiplier=1,
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DepthwiseConv2D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        if len(input_shape) < 4:
            raise ValueError('Inputs to `DepthwiseConv2D` should have rank 4. '
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`DepthwiseConv2D` '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        outputs = K.depthwise_conv2d(
            inputs,
            self.depthwise_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)

        if self.bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
            out_filters = input_shape[3] * self.depth_multiplier

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                             self.padding,
                                             self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                             self.padding,
                                             self.strides[1])

        if self.data_format == 'channels_first':
            return (input_shape[0], out_filters, rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, out_filters)

    def get_config(self):
        config = super(DepthwiseConv2D, self).get_config()
        config.pop('filters')
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['depth_multiplier'] = self.depth_multiplier
        config['depthwise_initializer'] = initializers.serialize(
            self.depthwise_initializer)
        config['depthwise_regularizer'] = regularizers.serialize(
            self.depthwise_regularizer)
        config['depthwise_constraint'] = constraints.serialize(
            self.depthwise_constraint)
        return config


def ConvBlock(kernels_size,filter_nums,data_in,**kargs):
    conv_stride = kargs.get('conv_stride',(1,1))
    w_initial = kargs.get('w_initial','glorot_uniform')
    w_decay = kargs.get('w_decay',l2(0.0005))
    momtum = kargs.get('momtum',0.9)
    eps = kargs.get('eps',1e-5)
    na = int(kargs.get('cb_name',1))
    train_bn = kargs.get('train_bn',True)
    conv_out = KL.Conv2D(filter_nums,kernels_size,strides=conv_stride,padding='same',use_bias=False,\
                kernel_initializer=w_initial,kernel_regularizer=w_decay,name='res0_conv_%d' % na)(data_in)
    bn_out = KL.BatchNormalization(momentum=momtum,epsilon=eps,name='res0_bn_%d' % na)(conv_out,training=train_bn)
    #out = keras.activations.relu(bn_out,max_value=6)
    out = KL.Activation('relu',name='res0_relu_%d' % na)(bn_out)
    return out

def Conv1x1(filter_nums,data_in,**kargs):
    is_linear = kargs.get('is_linear',False)
    w_initial = kargs.get('w_initial','glorot_uniform')
    w_decay = kargs.get('w_decay',l2(0.0005))
    momtum = kargs.get('momtum',0.9)
    eps = kargs.get('eps',1e-5)
    na = kargs.get('conv_name',"res1_1b_conv1a")
    train_bn = kargs.get('train_bn',True)
    conv_out = KL.Conv2D(filter_nums,(1,1),strides=(1,1),use_bias=False,\
                kernel_initializer=w_initial,kernel_regularizer=w_decay,name='%s_conv' % na)(data_in)
    bn_out = KL.BatchNormalization(momentum=momtum,epsilon=eps,name='%s_bn' % na)(conv_out,training=train_bn)
    if not is_linear:
        #out = keras.activations.relu(bn_out,max_value=6)
        #out = KL.ReLU(max_value=6,name='%s_relu' % na)(bn_out)
        out = KL.Activation('relu',name='%s_relu' % na)(bn_out)
    else:
        out = bn_out
    return out

def Dconv(data_in,**kargs):
    s = kargs.get('dconv_stride',1)
    w_initial = kargs.get('w_initial','glorot_uniform')
    w_decay = kargs.get('w_decay',l2(0.0005))
    momtum = kargs.get('momtum',0.9)
    eps = kargs.get('eps',1e-5)
    na = kargs.get('bot_name',"res1_1b")
    train_bn = kargs.get('train_bn',True)
    dconv_out = DepthwiseConv2D((3,3),strides=(s,s),padding='same',use_bias=False,depth_multiplier=1,\
                depthwise_initializer=w_initial,depthwise_regularizer=w_decay,name='%s_dconv' % na)(data_in)
    bn_out = KL.BatchNormalization(momentum=momtum,epsilon=eps,name='%s_dconv_bn' % na)(dconv_out,training=train_bn)
    #out = keras.activations.relu(bn_out,max_value=6)
    #out = KL.ReLU(max_value=6,name='%s_dconv_relu' % na)(bn_out)
    out = KL.Activation('relu',name='%s_dconv_relu' % na)(bn_out)
    return out

def bottleneck(chal_in,chal_out,data_in,**kargs):
    name = kargs.get('bot_name','res1_1b')
    name_child = name + "_conv1a"
    b_conv1_out = Conv1x1(chal_in,data_in,conv_name=name_child,**kargs)
    b_dconv_out = Dconv(b_conv1_out,**kargs)
    name_child = name + "_conv1b"
    b_conv2_out = Conv1x1(chal_out,b_dconv_out,is_linear=True,conv_name=name_child,**kargs)
    return b_conv2_out

def Inverted_residual_block(t,chal_in,c,s,data_in,**kargs):
    same_shape = kargs.get('same_shape',True)
    name = kargs.get('inv_block_name','res1_1')
    name_b = name+"b"
    rb_out = bottleneck(t*chal_in,c,data_in,dconv_stride=s,bot_name=name_b,**kargs)
    if s==1 :
        if not same_shape:
            name_a = name+"a"
            data_out = Conv1x1(c,data_in,conv_name=name_a,is_linear=True,**kargs)
            out = KL.add([rb_out,data_out])
        else:
            out = KL.add([rb_out,data_in])
    else:
        out = rb_out
    return out

def Inverted_residual_seq(t,chal_in,c,s,n,data_in,**kargs):
    name = kargs.get('name','res1')
    name_child = name + "_1"
    out = Inverted_residual_block(t,chal_in,c,s,data_in,inv_block_name=name_child,same_shape=False,**kargs)
    if n >1:
        for idx in range(1,n):
            name_child = name +"_%d" % (idx+1)
            out = Inverted_residual_block(t,chal_in,c,1,out,inv_block_name=name_child,**kargs)
    return out

def MobileNetV2(input_data,**kargs):
    width_mult = kargs.get('width_mult',1.0)
    class_num = kargs.get('class_num',81)
    cn = [int(x*width_mult) for x in [32,16,24,32,64,96,160,320,1280]]
    #input_data = KL.Input(tensor=input_tensor)
    #input_data = KL.Input(input_data)
    b0 = ConvBlock((3,3),cn[0],input_data,conv_stride=(2,2),**kargs)
    b1 = Inverted_residual_seq(1,cn[0],cn[1],1,1,b0,**kargs)
    b2 = Inverted_residual_seq(6,cn[1],cn[2],2,2,b1,name='res2',**kargs)
    b3 = Inverted_residual_seq(6,cn[2],cn[3],2,3,b2,name='res3',**kargs)
    b4 = Inverted_residual_seq(6,cn[3],cn[4],2,4,b3,name='res4',**kargs)
    b5 = Inverted_residual_seq(6,cn[4],cn[5],1,3,b4,name='res5',**kargs)
    b6 = Inverted_residual_seq(6,cn[5],cn[6],2,3,b5,name='res6',**kargs)
    b7 = Inverted_residual_seq(6,cn[6],cn[7],1,1,b6,name='res7',**kargs)
    b8 = ConvBlock((1,1),cn[8],b7,conv_stride=(1,1),cb_name=2,**kargs)
    #p1 = KL.GlobalAveragePooling2D()(b8)
    #fc = KL.Dense(class_num,use_bias=False,kernel_regularizer=l2(0.0005),name='fc1')(p1)
    #pred = KL.Activation('softmax',name='predictions')(fc)
    #net = Model(inputs=input_data,outputs=pred)
    return [b1,b2,b3,b4,b6]
    #return net

def vis_net(model):
    plot_model(model,show_shapes=True,show_layer_names=True,to_file='model_mobile.png')

def get_symbol(input_data,architecture,**kargs):
    #train_bn = kargs.get('train_bn',True)
    assert architecture in ["resnet50", "resnet101","mobilenet"]
    return MobileNetV2(input_data,**kargs)

if __name__ == '__main__':
    a = K.ones(shape=(1,640,640,3))
    in_d = KL.Input(tensor=a)
    net = get_symbol(in_d,'mobilenet')
    net_m = Model(input=in_d,output=net)
    #net = test()
    vis_net(net_m)