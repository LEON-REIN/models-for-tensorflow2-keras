# @.@ coding  : utf-8 ^_^
# @Author     : Leon Rein
# @Time       : 2022/8/11 ~ 上午10:55
# @File       : simple_models.py
# @Note       : Some simple decoders


import tensorflow as tf
from tensorflow.keras import layers
from .SE_ResNeXt_1DCNN import grouped_convolution_block


# region Decode 1D tensor to a 3D shape of output_shape
def simple_decoder_3D(latent_dims, output_shape, name="simple_decoder_3D"):
    out_H = int(output_shape[0])  # should be dividable by 4
    out_W = int(output_shape[1])  # should be dividable by 4
    out_C = int(output_shape[2])

    latent_inputs = layers.Input(shape=(latent_dims,))
    x = layers.Dense(out_H * out_W * 4, activation="relu")(latent_inputs)
    x = layers.Reshape((out_H // 4, out_W // 4, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    out = layers.Conv2DTranspose(out_C, 3, activation="sigmoid", padding="same")(x)

    return tf.keras.Model(latent_inputs, out, name=name)
# endregion


# region Decode 1D tensor to a 2D shape of output_shape
def conv_1D_block(x, filters, kernel_size, strides, activation=True):
    # 1D Convolutional Block with BatchNormalization
    x = layers.Conv1D(filters, kernel_size, strides=strides, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    if activation:
        x = layers.LeakyReLU(alpha=0.2)(x)
    return x


def resize_conv(x, filters, cardinality):
    x = layers.UpSampling1D(size=2)(x)
    x = conv_1D_block(x, filters=filters, kernel_size=3, strides=1)
    shortcut = x
    x = grouped_convolution_block(x, filters//2, 3, 1, cardinality)
    x = conv_1D_block(x, filters=filters, kernel_size=3, strides=1, activation=False)  # (None, 256, 64)
    return layers.Add()([x, shortcut])


def simple_decoder_2D(latent_dims, out_shape, out_activation=None, name="simple_decoder_2D"):
    # output_shape should be dividable by 16
    latent_inputs = layers.Input(shape=(latent_dims,))
    x = layers.Dense(out_shape[0]//16, activation=layers.LeakyReLU(alpha=0.2))(latent_inputs)
    x = tf.expand_dims(x, axis=-1)

    x = resize_conv(x, 64, 8)
    x = resize_conv(x, 32, 8)
    x = resize_conv(x, 16, 4)
    x = resize_conv(x, 16, 4)

    x = conv_1D_block(x, filters=out_shape[1]*2, kernel_size=3, strides=1)
    x = conv_1D_block(x, filters=out_shape[1], kernel_size=3, strides=1, activation=False)
    if out_activation is not None:
        x = out_activation(x)
    return tf.keras.Model(inputs=latent_inputs, outputs=x, name=name)
# endregion


def simple_CNN_1D(input_shape, out_dim, out_activation=None):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = layers.Conv1D(8, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.AveragePooling1D()(x)
    x = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.AveragePooling1D()(x)
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.AveragePooling1D()(x)
    x = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.AveragePooling1D()(x)
    x = layers.Conv1D(1, kernel_size=1, activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(out_dim, activation=out_activation)(x)
    return tf.keras.Model(inputs, x, name="CNN 1D Encoder")

if __name__ == '__main__':
    latent_dim = 256
    dec = simple_decoder_3D(latent_dim, (28, 28, 1))
    dec2 = simple_decoder_2D(latent_dim, (20000, 2), out_activation=None)
    dec2.summary()
