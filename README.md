# In short
my models for tensorflow2::keras, e.g., Gradient Reversal Layer, ResNeXt1D, py_function decorator, etc.

- Attention!
  - My main ***references*** are listed at the beginning of each file, to which I am very grateful.
  - How-tos are listed at the bottom of the file.

# `SE_ResNeXt_1DCNN.py`
## Summary
1D SE-ResNeXt models for Keras via pure functional API by tf.keras.

## How to
```python
# Configurations
params = {"length": 3000,  # length of input series
      "num_channels": 2,  # channel (last) of input series
      "num_filters": 16,  # Width of the Initial Layer, subsequent layers start from here
      "cardinality": 8,
      "output_nums": 4,  # Number of Class for Classification Problems, always '1' for Regression Problems
      "pooling": 'avg',
      "dropout_rate": 0.5}

# Build, Compile and Print Summary
my_model = SEResNeXt(**params).SEResNeXt50()
Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
              loss=tf.keras.losses.MeanAbsoluteError(),
              metrics=tf.keras.metrics.MeanSquaredError())
Model.summary()
```

# `domain_adversarial_model.py`
## Summary
Keras implementation of a gradient reversal layer (GRL), and a simple model of Domain-Adversarial Training of Neural Networks (DANN).

## How to
```python
# Firstly, you should have a feature extraction model of tf.keras.Model, e.g., a tf.keras.applications.ResNet50 with `include_top=false`.
from .SE_ResNeXt_1DCNN import SEResNeXt 
Model = SEResNeXt(...).SEResNeXt50()  # A custom model to be specified by yourself

# If the model doesn't output a feature, several top layers have to be removed like:
modified_model = tf.keras.Model(inputs=Model.input, outputs=Model.get_layer(index=-2).output)
DANN = DomainAdversarialModel(feature_extractor=modified_model, num_labels=4, num_domains=3, lambda_=-0.8).get_model()
DANN.summary()
DANN.compile(...)   # To be specified by yourself

data = tf.random.normal((1, ...))  # To be specified by yourself
print(DANN(data))
```

# `data_mapper.py`
## Summary
Two decorators that wraps a normal python to tf.py_function or tf.numpy_function that avoiding erros like `AttributeError: ‘Tensor‘ object has no attribute ‘numpy‘`. This may happen in your data processing function to tf.data.Dataset.map().
> See also: https://blog.csdn.net/qq_27329407/article/details/125722048

## How to 
The two decorators only have one different and `input_signature` can be removed easily by yourself.
```python
# Example of loading arrays from many txt files:
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
```

# `losses_metrics.py `
## Summary
Some examples of making custom loss functions and metrics. To be improved.
