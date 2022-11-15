# @.@ coding  : utf-8 ^_^
# @Author     : Leon Rein
# @Time       : 2022/8/5 ~ 下午8:05
# @File       : variational_autoencoder.py
# @Note       : A implementation of Variational AutoEncoder.
# @References : https://keras.io/examples/generative/vae/
# @Attention  : tf.__version__ >= 2.4.0 (maybe OK at 2.3.0)

import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import keras.backend as K


class SamplingLayer(tf.keras.layers.Layer):
    """
    Generates a gaussian distribution over the mean and log-variance from the inputs.
    """

    def call(self, inputs, *args, **kwargs):
        """
        It takes the mean and log-variance of the latent space and randomly samples from it

        Args:
          inputs: A tuple of tensors, where the first tensor is the mean and the second tensor is the log-variance.

        Returns:
          The re-parameterization trick is used to sample from a distribution by transforming a random uniform variable,
          so that the sampling process can pass the gradients to the inputs
        """
        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))  # normal distribution N(0, I)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon  # also a gaussian distribution

    def compute_output_shape(self, input_shape):
        return input_shape[0]  # input_shape = [(None, latent_dim), (None, latent_dim)]


class Scaler(tf.keras.layers.Layer):
    """
    A layer after BN to z_mean (also z_log_var) to prevent to vanish of KL loss
    References: https://spaces.ac.cn/archives/7381
    """

    def __init__(self, tau=0.5, **kwargs):
        super(Scaler, self).__init__(**kwargs)
        self.scale = None
        self.tau = tau

    def build(self, input_shape):
        super(Scaler, self).build(input_shape)
        self.scale = self.add_weight(
            name='scale', shape=(input_shape[-1],), initializer='zeros'
        )

    def call(self, inputs, mode='positive'):
        if mode == 'positive':
            scale = self.tau + (1 - self.tau) * tf.keras.backend.sigmoid(self.scale)
        else:
            scale = (1 - self.tau) * tf.keras.backend.sigmoid(-self.scale)
        return inputs * tf.keras.backend.sqrt(scale)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'tau': self.tau
        })
        return config


class DifferenceFromMean(tf.keras.layers.Layer):
    """这是个简单的层，定义q(z|y)中的均值参数，每个类别配一个均值。
    然后输出“z - 均值”，为后面计算loss准备。
    """
    def __init__(self, num_classes, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.mean = self.add_weight(name='mean',
                                    shape=(num_classes, latent_dim),
                                    initializer='zeros')

    def call(self, inputs):
        z = inputs # z.shape=(batch_size, latent_dim)
        z = K.expand_dims(z, 1)
        return z - K.expand_dims(self.mean, 0)

    def compute_output_shape(self, input_shape):
        return None, self.num_classes, input_shape[-1]


# noinspection PyAbstractClass,PyMethodOverriding
class VariationalAutoEncoder(tf.keras.Model):
    def __init__(self, encoder, decoder, latent_dim: int = 2, recon_weight: float = 1.0, use_bias=True, use_BN=False,
                 **kwargs):
        """
        The basic implementation of a VAE model.
        It will wrap the encoder to latent features (z_mean, z_log_var), and also add a KL loss to it.
        Customize the encoder at encoder_wrapper() if necessary.

        Args:
          encoder: The encoder model, it should output a flattened representation (feature). And it must be Functional
            (created by tf.keras.Model), it must have attributes: .input and .output
          decoder: The decoder model of Functional (created by tf.keras.Model). Its input shape should be (latent_dim,)
          latent_dim: Dimension of the latent features
          use_BN: Whether to use Batch Normalization to z_mean and z_log_var to prevent the KL loss vanishing.
                  May not work. References: https://spaces.ac.cn/archives/7381
          use_bias: Whether to use bias in the linear layers of generating the z_mean and z_log_var
          recon_wight: The weight of the reconstruction loss. Recon & KL Losses are evaluated after taking the mean.

        Notes:
          You can just get the decoder via vae_obj.decoder, so does the encoder.
          If the encoder or decoder has its additional losses, they're already taken into consideration.
          The whole model can be reused in other models' training steps.

        """
        super(VariationalAutoEncoder, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.use_bias = use_bias
        self.use_BN = use_BN
        self.recon_weight = recon_weight
        # metrics
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        # encoder & decoder
        self.encoder = self.encoder_wrapper(encoder)  # wrap the encoder
        self.decoder = decoder
        # defined by .compile()
        self.recon_loss_func = None
        self.optimizer = None

    def compile(self, optimizer, loss, **kwargs):
        super().compile(**kwargs)
        self.optimizer = tf.keras.optimizers.Adam(0.001) if optimizer == 'adam' else optimizer
        self.recon_loss_func = tf.keras.losses.binary_crossentropy if loss == 'bce' else \
            tf.keras.losses.mse if loss == 'mse' else loss

    @property
    def metrics(self):

        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def encoder_wrapper(self, encoder):
        """
        It takes in an encoder model and returns a new encoder model, which will generate a latent feature and have a
        KL loss. The mean and log-var each are generated by a linear layer, then resampled by the SamplingLayer.

        Args:
          encoder: The encoder model of Functional (created by tf.keras.Model), it must have attributes: .input and
            .output

        Returns:
          The encoder model is being returned.
        """

        z_mean = tf.keras.layers.Dense(self.latent_dim, name="z_mean", use_bias=self.use_bias)(encoder.output)
        z_log_var = tf.keras.layers.Dense(self.latent_dim, name="z_log_var", use_bias=self.use_bias)(encoder.output)

        if self.use_BN:
            scaler = Scaler()
            z_mean = tf.keras.layers.BatchNormalization(scale=False, center=False, epsilon=1e-8,
                                                        trainable=True)(z_mean)
            z_mean = scaler(z_mean, mode='positive')
            z_log_var = tf.keras.layers.BatchNormalization(scale=False, center=False, epsilon=1e-8,
                                                           trainable=True)(z_log_var)
            z_log_var = scaler(z_log_var, mode='negative')

        # Sampling Layer!!
        z = SamplingLayer()([z_mean, z_log_var])
        enc = tf.keras.Model(encoder.input, [z_mean, z_log_var, z], name="encoder")
        # add loss
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))  # shape=(batch_size, latent_dim)
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        enc.add_loss(kl_loss)  # self.encoder.losses[0]
        return enc

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            self(data, training=True)  # losses are logged in self.losses including every extra loss

        '''self.losses: including every sub-layer's extra added loss through add_loss() ;
        self.trainable_weights: also including all, i.e. encoder and decoder's'''
        grads = tape.gradient(self.losses, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # How to get the losses from self.losses
        self.total_loss_tracker.update_state(sum(self.losses))
        self.reconstruction_loss_tracker.update_state(self.losses[0])
        self.kl_loss_tracker.update_state(self.losses[1])

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, inputs, training=None, mask=None):
        """Go to customized train_step to see how this model can be reused and get its losses in other models."""
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        if not training:
            return z_mean
        batch_size = tf.cast(tf.shape(inputs)[0], tf.float32)
        reconstruction = self.decoder(z, training=True)
        reconstruction_loss = \
            tf.reduce_sum(self.recon_loss_func(inputs, reconstruction)) / batch_size * self.recon_weight
        # reconstruction_loss = tf.reduce_mean(
        #     tf.reduce_sum(
        #         keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
        #     )
        # )
        self.add_loss(reconstruction_loss)  # can be accessed by self.losses
        return z

    def plot_label_clusters(self, data, labels):
        # display a 2D plot of the digit classes in the latent space
        z_mean, _, _ = self.encoder.predict(data)
        if self.latent_dim != 2:
            z_mean = TSNE(2, init='pca', learning_rate='auto', verbose=1).fit_transform(z_mean)
        plt.figure(figsize=(16, 14))
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.show()

    def plot_generation_results(self, num=30, fig_size=20):
        # for mnist dataset with latent dimension 2, may not work properly for other datasets
        if self.latent_dim != 2:
            raise NotImplementedError('Condition of latent_dim != 2 is not implemented yet!')

        digit_size = 28
        scale = 1.0
        figure = np.zeros((digit_size * num, digit_size * num))
        grid_x = np.linspace(-scale, scale, num)
        grid_y = np.linspace(-scale, scale, num)[::-1]
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = self.decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(fig_size, fig_size))
        start_range = digit_size // 2
        end_range = num * digit_size + start_range
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap="Greys_r")
        plt.show()


if __name__ == '__main__':
    """A Simple Training Example!"""
    import numpy as np

    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
    print(mnist_digits.shape)

    """Simple encoder and decoder"""
    '''Encoder'''
    encoder_inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    default_encoder = tf.keras.Model(encoder_inputs, x, name="encoder")
    '''Decoder'''
    latent_dims = 32
    latent_inputs = tf.keras.layers.Input(shape=(latent_dims,))
    x = tf.keras.layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = tf.keras.layers.Reshape((7, 7, 64))(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = tf.keras.layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    default_decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")

    """VAE"""
    # In this case, the KL loss will vanish to 0 when use_BN=False, recon_weight=1, and loss='mse';
    vae = VariationalAutoEncoder(encoder=default_encoder, decoder=default_decoder,
                                 latent_dim=latent_dims,
                                 recon_weight=5,  # weight may reach higher ~
                                 use_BN=False)
    vae.compile(optimizer='adam', loss='mse', run_eagerly=False)  # mse or bce

    # training
    vae.fit(mnist_digits, epochs=51, batch_size=256)
    # plot label clusters in latent space (z_mean)
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train[:9000], -1).astype("float32") / 255
    vae.plot_label_clusters(x_train[:9000], y_train[:9000])
    # plot generation results from latent space
    # vae.plot_generation_results()  # only implementation for latent dimension 2!
