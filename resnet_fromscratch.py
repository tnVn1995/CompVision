"""Implementing Resnet frmo scratch using functional API"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Activation, Dense, Flatten, Conv2D, MaxPooling2D, 
    GlobalAveragePooling2D, AveragePooling2D, BatchNormalization, add)
import tensorflow.keras.regularizers as regulizers

#%% Res Conv layer
def _res_conv(filters, kernel_size=3, padding='same', strides=1, use_relu=True, use_bias=False, name='cbr',
              kernel_initializer='he_normal', kernel_regularizer=regulizers.l2(1e-4)):
    """[Return a layer block chaining conv, batchnrom and reLU activation.]
    
    Arguments:
        filters {[int]} -- [Number of filters]
    
    Keyword Arguments:
        kernel_size {int} -- [Kernel size.] (default: {3})
        padding {str} -- [Convolution padding.] (default: {'same'})
        strides {int} -- [ Convolution strides.] (default: {1})
        use_relu {bool} -- [Flag to apply ReLu activation at the end.] (default: {True})
        use_bias {bool} -- [Flag to use bias or not in Conv layer.] (default: {False})
        name {str} -- [Name suffix for the layers.] (default: {'cbr'})
        kernel_initializer {str} -- [Kernel initialization method name.] (default: {'he_normal'})
        kernel_regularizer {[type]} -- [ Kernel regularizer.] (default: {regulizers.l2(1e-4)})
    
    Returns:
        [type] -- [Callable layer block]
    """  

    def layer_fn(x):
        conv = Conv2D(
            filters=filters, kernel_size=kernel_size, padding=padding, strides=strides, use_bias=use_bias,
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, 
            name=name + '_c')(x)
        res = BatchNormalization(axis=-1, name=name + '_bn')(conv)
        if use_relu:
            res = Activation("relu", name=name + '_r')(res)
        return res

    return layer_fn

#%%  Merge 

def _merge_with_shortcut(kernel_initializer='he_normal', kernel_regularizer=regulizers.l2(1e-4), 
                         name='block'):
    """[Return a layer block which merge an input tensor and the corresponding 
    residual output tensor from another branch.]
    
    Keyword Arguments:
        kernel_initializer {str} -- [Kernel initialisation method name.] (default: {'he_normal'})
        kernel_regularizer {[type]} -- [Kernel regularizer.] (default: {regulizers.l2(1e-4)})
        name {str} -- [Name suffix for the layers.] (default: {'block'})
    
    Returns:
        [type] -- [Callable layer block]
    """ 

    def layer_fn(x, x_residual):
        # In the original paper, the input and the residual output must have the same dimensions
        # for element-wise addition. We'll use a conv (1x1) to match dimensions if they are not equal
        x_shape = tf.keras.backend.int_shape(x)
        x_residual_shape = tf.keras.backend.int_shape(x_residual)
        if x_shape == x_residual_shape:
            shortcut = x
        else:
            strides = (
                int(round(x_shape[1] / x_residual_shape[1])), # vertical stride
                int(round(x_shape[2] / x_residual_shape[2]))  # horizontal stride
            )
            x_residual_channels = x_residual_shape[3]
            shortcut = Conv2D(
                filters=x_residual_channels, kernel_size=(1, 1), padding="valid", strides=strides,
                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
                name=name + '_shortcut_c')(x)

        merge = add([shortcut, x_residual])
        return merge

    return layer_fn

#%% Residual Block

def _residual_block_basic(filters, kernel_size=3, strides=1, use_bias=False, name='res_basic',
                          kernel_initializer='he_normal', kernel_regularizer=regulizers.l2(1e-4)):
    """[Return a basic residual layer block.]
    
    Arguments:
        filters {[type]} -- [Number of filters.]
    
    Keyword Arguments:
        kernel_size {int} -- [Kernel size.] (default: {3})
        strides {int} -- [Convolution strides] (default: {1})
        use_bias {bool} -- [Flag to use bias or not in Conv layer.] (default: {False})
        name {str} -- [name of the layer.] (default: {'res_basic'})
        kernel_initializer {str} -- [Kernel initialisation method name.] (default: {'he_normal'})
        kernel_regularizer {[type]} -- [Kernel regularizer.] (default: {regulizers.l2(1e-4)})
    
    Returns:
        [type] -- [Callable layer block]
    """     
    def layer_fn(x):
        x_conv1 = _res_conv(
            filters=filters, kernel_size=kernel_size, padding='same', strides=strides, 
            use_relu=True, use_bias=use_bias,
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
            name=name + '_cbr_1')(x)
        x_residual = _res_conv(
            filters=filters, kernel_size=kernel_size, padding='same', strides=1, 
            use_relu=False, use_bias=use_bias,
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
            name=name + '_cbr_2')(x_conv1)
        merge = _merge_with_shortcut(kernel_initializer, kernel_regularizer,name=name)(x, x_residual)
        merge = Activation('relu')(merge)
        return merge

    return layer_fn

#%% Bottle_kneck - Optional for Deeper Network

# def _residual_block_bottleneck(filters, kernel_size=3, strides=1, use_bias=False, name='res_bottleneck',
#                                kernel_initializer='he_normal', kernel_regularizer=regulizers.l2(1e-4)):
#     """[Return a residual layer block with bottleneck, recommended for deep ResNets 
#     (depth > 34).]
    
#     Arguments:
#         filters {[type]} -- [Number of filters.]
    
#     Keyword Arguments:
#         kernel_size {int} -- [Kernel size.] (default: {3})
#         strides {int} -- [Convolution strides] (default: {1})
#         use_bias {bool} -- [Flag to use bias or not in Conv layer.] (default: {False})
#         name {str} -- [name of the layer] (default: {'res_bottleneck'})
#         kernel_initializer {str} -- [Kernel initialisation method name.] (default: {'he_normal'})
#         kernel_regularizer {[type]} -- [Kernel regularizer.] (default: {regulizers.l2(1e-4)})
    
#     Returns:
#         [type] -- [Callable layer block]
#     """    


#     def layer_fn(x):
#         x_bottleneck = _res_conv(
#             filters=filters, kernel_size=1, padding='valid', strides=strides, 
#             use_relu=True, use_bias=use_bias,
#             kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
#             name=name + '_cbr1')(x)
#         x_conv = _res_conv(
#             filters=filters, kernel_size=kernel_size, padding='same', strides=1, 
#             use_relu=True, use_bias=use_bias,
#             kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
#             name=name + '_cbr2')(x_bottleneck)
#         x_residual = _res_conv(
#             filters=filters * 4, kernel_size=1, padding='valid', strides=1, 
#             use_relu=False, use_bias=use_bias,
#             kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
#             name=name + '_cbr3')(x_conv)
#         merge = _merge_with_shortcut(kernel_initializer, kernel_regularizer, name=name)(x, x_residual)
#         merge = Activation('relu')(merge)
#         return merge

#     return layer_fn


#%%  Chaining blocks together

def _residual_macroblock(block_fn, filters, repetitions=3, kernel_size=3, strides_1st_block=1, use_bias=False,
                         kernel_initializer='he_normal', kernel_regularizer=regulizers.l2(1e-4),
                         name='res_macroblock'):
    """[Return a layer block, composed of a repetition of `N` residual blocks.]
    
    Arguments:
        block_fn {[type]} -- [Block layer method to be used.]
        filters {[type]} -- [Number of filters.]
    
    Keyword Arguments:
        repetitions {int} -- [Number of times the block should be repeated inside.] (default: {3})
        kernel_size {int} -- [Kernel size.] (default: {3})
        strides_1st_block {int} -- [Convolution strides for the 1st block.] (default: {1})
        use_bias {bool} -- [Flag to use bias or not in Conv layer.] (default: {False})
        kernel_initializer {str} -- [Kernel initialisation method name.] (default: {'he_normal'})
        kernel_regularizer {[type]} -- [Kernel regularizer.] (default: {regulizers.l2(1e-4)})
        name {str} -- [Callable layer block] (default: {'res_macroblock'})
    """                         

    def layer_fn(x):
        for i in range(repetitions):
            block_name = "{}_{}".format(name, i) 
            strides = strides_1st_block if i == 0 else 1
            x = block_fn(filters=filters, kernel_size=kernel_size, 
                         strides=strides, use_bias=use_bias,
                         kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
                         name=block_name)(x)
        return x

    return layer_fn