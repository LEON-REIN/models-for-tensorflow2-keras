# @.@ coding  : utf-8 ^_^
# @Author     : Leon Rein
# @Time       : 2022/6/12 ~ 下午7:25
# @File       : data_mapper.py
# @Note       : 一些数据映射的函数, 输出的类型为 tf 的数据类型. Channel last!!!
from typing import Union
import tensorflow as tf


@tf.function
def complex_mapping(sample, label):
    """
    It takes a complex-valued sample and returns a real-valued tensor with its labels

    .. warning:: Data returned with channel **last** format. SISO!!

    Args:
      sample: the input data
      label: The labels of the sample.

    Returns:
      The sample and 2 labels (TX, RX) are being returned.
    """
    # Normalization
    sample = (sample - tf.math.reduce_mean(sample)) / tf.cast(tf.math.reduce_std(sample), tf.complex128)
    # Concatenate the real part and the imaginary part
    sample = tf.concat([tf.expand_dims(tf.math.real(sample), -1), tf.expand_dims(tf.math.imag(sample), -1)],
                       axis=-1)
    # tf.keras.utils.to_categorical()
    return tf.cast(sample, tf.float32), tf.cast(label, tf.uint8)


@tf.function
def complex_mapping_noRX(sample, label):
    """
    It takes a complex-valued sample and returns a real-valued tensor and the first element of given labels

    .. warning:: Data returned with channel **last** format.

    Args:
      sample: the input data
      label: The label of the sample.

    Returns:
      The sample and 1 labels (TX) are being returned.
    """
    # Normalization
    sample = (sample - tf.math.reduce_mean(sample)) / tf.cast(tf.math.reduce_std(sample), tf.complex128)
    # Concatenate the real part and the imaginary part
    sample = tf.concat([tf.expand_dims(tf.math.real(sample), -1), tf.expand_dims(tf.math.imag(sample), -1)],
                       axis=-1)
    # tf.keras.utils.to_categorical()
    return tf.cast(sample, tf.float32), tf.cast(label[0], tf.uint8)


@tf.function
def complex_mapping_short(sample, label):
    """
    It takes a complex sample, normalizes it, and then concatenates the real and imaginary parts into a single tensor

    Args:
      sample: the raw data
      label: The label of the sample.

    Returns:
      The sample and label are being returned.
    """
    sample = sample[:8192]
    sample = (sample - tf.math.reduce_mean(sample)) / tf.cast(tf.math.reduce_std(sample), tf.complex128)
    # Concatenate the real part and the imaginary part
    sample = tf.concat([tf.expand_dims(tf.math.real(sample), -1), tf.expand_dims(tf.math.imag(sample), -1)],
                       axis=-1)
    return tf.cast(sample, tf.float32), tf.cast(label, tf.uint8)


@tf.function
def stft_mapping():
    pass


def naive_function(input_signature: Union[tuple, list], type_out: Union[tuple, list]):
    """
    Wraps a python function into a TensorFlow op that executes it eagerly.

    Args:
      input_signature: This is the input signature of the function. It is a list of TensorSpec objects.
      type_out: The output types of the function.

    Returns:
      A decorator function
    """
    if type_out is None:
        raise TypeError("You must provide output types as a list or a tuple!")

    def map_decorator(func):
        @tf.function(input_signature=input_signature)
        def wrapper(*args):
            # Use a tf.py_function to prevent auto-graph from compiling the method
            return tf.py_function(
                func,
                inp=args,
                Tout=type_out
            )

        return wrapper

    return map_decorator


def np_function(input_signature: Union[tuple, list], type_out: Union[tuple, list]):
    """
    It takes a function and returns a function that takes a tensor as input and returns a tensor as output

    Args:
      input_signature: This is the input signature of the function. It is a list of TensorSpec objects.
      type_out (Union[tuple, list]): The output types of the function.

    Returns:
      A decorator that takes a function and returns a wrapper function.
    """
    if type_out is None:
        raise TypeError("You must provide output types as a list or a tuple!")

    def map_decorator(func):

        @tf.function(input_signature=input_signature)
        def wrapper(*args):
            return tf.numpy_function(func, inp=args, Tout=type_out)

        return wrapper

    return map_decorator


if __name__ == '__main__':
    import numpy as np

    # 两个装饰器的用法

    @naive_function(input_signature=tf.TensorSpec(shape=None, dtype=tf.string), type_out=[tf.float32, tf.uint8])
    def load_data(file_name):  # get eager tensors here
        file_name = file_name.numpy().decode("utf8")
        label = ...
        data = np.loadtxt(file_name)[..., np.newaxis]
        return tf.cast(data, tf.float32), tf.cast(label, tf.uint8)

    @np_function(input_signature=tf.TensorSpec(shape=None, dtype=tf.string), type_out=[tf.float32, tf.uint8])
    def load_data(file_name):  # get numpy types here
        file_name = file_name.decode("utf8")
        label = ...
        data = np.loadtxt(file_name)[..., np.newaxis]
        return tf.cast(data, tf.float32), tf.cast(label, tf.uint8)
