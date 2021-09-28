import tensorflow as tf
import numpy as np

content_layer = ['conv2d_88']
style_layers = ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4']

def inception_model(layer_names):
    inception = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False)
    inception.trainable = False
    output_layers = [inception.get_layer(layer_name).output for layer_name in layer_names]
    return tf.keras.models.Model(inputs=inception.input, outputs=output_layers)

inception = inception_model(style_layers+content_layer)

def get_style_loss(features, targets):
    return tf.reduce_mean(tf.square(features-targets))

def get_content_loss(features, targets):
    return 0.5*tf.reduce_sum(tf.square(features-targets))

def gram_matrix(input_tensor):
    gram = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    height = input_shape[1]
    width = input_shape[2]
    num_locations = tf.cast(height * width, tf.float32)
    return gram/num_locations




