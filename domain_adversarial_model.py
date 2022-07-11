# References:
#   https://stackoverflow.com/questions/56841166/how-to-implement-gradient-reversal-layer-in-tf-2-0
#   https://www.tensorflow.org/guide/eager#%E8%87%AA%E5%AE%9A%E4%B9%89%E6%A2%AF%E5%BA%A6

import tensorflow as tf


class GradientReversalLayer(tf.keras.layers.Layer):
    """The gradient reversal layer is a layer that multiplies the gradient by a negative constant during
    backpropagation.

    Args:
      lambda_: Float32, the constant by which the gradient is multiplied. It should be a negative number.

    References:
      https://stackoverflow.com/questions/56841166/how-to-implement-gradient-reversal-layer-in-tf-2-0
      https://www.tensorflow.org/guide/eager#%E8%87%AA%E5%AE%9A%E4%B9%89%E6%A2%AF%E5%BA%A6
    """
    def __init__(self, lambda_: float = -1):
        super().__init__(trainable=False, name="gradient_reversal_layer")
        self.lambda_ = tf.constant(lambda_, dtype=tf.float32)  # Normally, a negative value

    def call(self, x, **kwargs):
        return self.grad_reversed(x)

    @tf.custom_gradient
    def grad_reversed(self, x):
        """
        It returns input and a custom gradient function.

        Args:
          x: The input tensor.

        Returns:
          the input x and the custom gradient function.
        """
        def custom_gradient(dy):
            return self.lambda_ * dy

        return x, custom_gradient

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "lambda": float(self.lambda_.numpy())
        })
        return config


class DomainAdversarialModel:
    """
    Domain-Adversarial Training of Neural Networks (DANN) in Tensorflow2.

    Args:
      feature_extractor: A model of tf.keras.Model, it would have attributes like .input, .output
      num_labels: Int, the number of labels.
      num_domains: Int, the number of domains.
      lambda_: Float32, the constant by which the gradient is multiplied. It should be a negative number.

    Attributes:
      output layer name of label classifier: "label_predict"
      output layer name of domain classifier: "domain_predict"
    """

    def __init__(self, feature_extractor: tf.keras.Model, num_labels, num_domains, lambda_: float = -1,
                 name_label_classifier="label_predict", name_domain_classifier="domain_predict"):
        self.feature_extractor = feature_extractor  # has to be a tf.keras.Model
        self.num_labels = num_labels
        self.num_domains = num_domains
        self.lambda_ = lambda_
        self.name_label_classifier = name_label_classifier
        self.name_domain_classifier = name_domain_classifier

    def get_model(self):
        feature = self.feature_extractor.output
        if len(self.feature_extractor.output_shape) != 2:
            # make sure feature has a shape of (None, feature_dim). Flatten is important for pytorch, maybe
            # not necessary for Tensorflow2.keras.
            feature = tf.keras.layers.Flatten()(feature)

        # output1 --> label_classifier
        label_predict = self.label_classifier(feature)
        # output2 --> domain_classifier
        domain_predict = GradientReversalLayer(self.lambda_)(feature)
        domain_predict = self.domain_classifier(domain_predict)

        return tf.keras.Model(inputs=self.feature_extractor.input,
                              outputs=[label_predict, domain_predict])

    def label_classifier(self, x):
        # x = tf.keras.layers.Dense(128, activation='relu')(x)
        return tf.keras.layers.Dense(self.num_labels, activation='softmax', name=self.name_label_classifier)(x)

    def domain_classifier(self, x):
        # x = tf.keras.layers.Dense(128, activation='relu')(x)
        return tf.keras.layers.Dense(self.num_domains, activation='softmax', name=self.name_domain_classifier)(x)


if __name__ == '__main__':
    """How to use DomainAdversarialModel"""
    # Firstly, you should have a feature extraction model of tf.keras.Model.
    from .SE_ResNeXt_1DCNN import SEResNeXt 
    Model = SEResNeXt(...).SEResNeXt50()  # To be specified by yourself

    # If the model doesn't output a feature, several top layers have to be removed like:
    modified_model = tf.keras.Model(inputs=Model.input, outputs=Model.get_layer(index=-2).output)
    DANN = DomainAdversarialModel(feature_extractor=modified_model, num_labels=4, num_domains=3, lambda_=-0.8).get_model()
    DANN.summary()
    DANN.compile(...)   # To be specified by yourself
    data = tf.random.normal((1, ...))  # To be specified by yourself
    print(DANN(data))
  
