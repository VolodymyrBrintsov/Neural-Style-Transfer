import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

content_layer = ['conv2d_88']
style_layers = ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4']


def tensor_to_image(tensor):
    #converts a tensor to an image
    tensor_shape = tf.shape(tensor)
    number_elem_shape = tf.shape(tensor_shape)
    if number_elem_shape > 3:
        assert tensor_shape[0] == 1
        tensor = tensor[0]
    return tf.keras.preprocessing.image.array_to_img(tensor)


def load_img(path_to_img):
    #loads an image as a tensor and scales it to 512 pixels
    max_dim = 512
    image = tf.io.read_file(path_to_img)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)

    shape = tf.shape(image)[:-1]
    shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    image = tf.image.resize(image, new_shape)
    image = image[tf.newaxis, :]
    image = tf.image.convert_image_dtype(image, tf.uint8)

    return image


def load_images(content_path, style_path):
    #loads the content and path images as tensors
    content_image = load_img("{}".format(content_path))
    style_image = load_img("{}".format(style_path))
    return content_image, style_image


def imshow(image, title=None):
    #displays an image with a corresponding title
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


def show_images_with_objects(images, titles=[]):
    #displays a row of images with corresponding titles
    if len(images) != len(titles):
        return

    plt.figure(figsize=(20, 12))
    for idx, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), idx + 1)
        plt.xticks([])
        plt.yticks([])
        imshow(image, title)


def clip_image_values(image, min_value=0.0, max_value=255.0):
    #clips the image pixel values by the given min and max
    return tf.clip_by_value(image, clip_value_min=min_value, clip_value_max=max_value)


def preprocess_image(image):
    #preprocesses a given image to use with Inception model
    image = tf.cast(image, dtype=tf.float32)
    image = (image / 127.5) - 1.0
    return image

content_path = tf.keras.utils.get_file('content_image.jpg','https://storage.googleapis.com/laurencemoroney-blog.appspot.com/MLColabImages/dog1.jpg')
style_path = tf.keras.utils.get_file('style_image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')
content_image, style_image = load_images(content_path, style_path)

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

def get_style_image_feature(image):
    preprocessed_style_image = preprocess_image(image)
    outputs = inception(preprocessed_style_image)
    style_outputs = outputs[:5]
    gram_style_features = [gram_matrix(style_output) for style_output in style_outputs]
    return gram_style_features

def get_content_image_features(image):
    preprocessed_content_image = preprocess_image(image)
    outputs = inception(preprocessed_content_image)
    content_outputs = outputs[5:]
    return content_outputs

def get_style_content_loss(style_targets, style_outputs, ):
    pass






