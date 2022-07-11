# @.@ coding  : utf-8 ^_^
# @Author     : Leon Rein
# @Time       : 2022/6/13 ~ 下午7:30
# @File       : losses_metrics.py
# @Note       : 自定义的损失函数和指标

import tensorflow as tf
from tensorflow.keras import losses, metrics


class MultiCrossEntropy(losses.Loss):
    """
    双任务的损失, y_true shape 暂时固定为 (None, 2), 可以对标签的第 task 个维度做交叉熵计算.

    Args:
      clipped_idx: Int, 损失计算截止的下标, 标签大于此数的样本的损失视为 0
    """

    def __init__(self, task: int, num_classes: int, label_smoothing: float = 0,
                 clipped_idx: int = None):
        super(MultiCrossEntropy, self).__init__()
        self.task = task  # 0/1, which column is used to compute the loss
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.clipped_idx = clipped_idx

    @tf.function
    def call(self, y_true, y_pred):
        # print(y_true.shape, y_pred.shape)  # (None, 2) (None, 4)
        if self.clipped_idx is not None:  # Only take source domain indexes into account
            condition = tf.where(y_true[:, 1-self.task] < self.clipped_idx)
            y_true = tf.gather(y_true[:, self.task], condition)  # (None, 1)
            y_pred = tf.gather(y_pred, condition)  # (None, 4)
            # print(y_true.shape, y_pred.shape)
        else:
            y_true = y_true[:, self.task]  # (None, 1)
        return losses.categorical_crossentropy(tf.one_hot(y_true, depth=self.num_classes), y_pred,
                                               label_smoothing=0)


class MultiAccuracy:
    """取出联合标签中的第 task 个元素求 ACC, 在 model.fit 时接收一个 batch 的数据; 离线使用需要放入一个 batch 中不然形状冲突"""
    def __init__(self, task):
        self.task = task

    def multi_sparse_categorical(self, y_true, y_pred):
        y_true = y_true[:, self.task]
        return tf.reduce_mean(metrics.sparse_categorical_accuracy(y_true, y_pred))
