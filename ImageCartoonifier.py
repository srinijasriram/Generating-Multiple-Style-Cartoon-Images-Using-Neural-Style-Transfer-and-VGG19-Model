import os
import tensorflow as tf
import keras
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from StyleContentModel import StyleContentModel

os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'


def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def style_loss_mean(name, style_outputs, style_targets):
    mean_style_loss = 0
    num_styles = len(style_targets)
    for style_target in style_targets:
        mean_style_loss += (style_outputs[name] - style_target[name])
    return mean_style_loss / num_styles


def style_content_loss(outputs, style_targets, num_style_layers, content_targets,
                       content_weight, num_content_layers, style_weight):
    style_outputs = outputs['style']
    content_outputs = outputs['content']

    style_loss = tf.add_n([tf.reduce_mean(
        (style_loss_mean(name, style_outputs, style_targets)) ** 2)
        for name in style_outputs.keys()])
     
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss


class Cartoonifier:
    mpl.rcParams['figure.figsize'] = (12, 12)
    mpl.rcParams['axes.grid'] = False
    style_weight = 1000
    content_weight = 1e4
    total_variation_weight = 30
    opt = keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    def __init__(self, content_path, style_paths):
        content_layers = ['block5_conv2']
        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1',
                        'block4_conv1',
                        'block5_conv1']
        self.extractor = StyleContentModel(style_layers, content_layers)
        self.num_style_layers = len(style_layers)
        self.num_content_layers = len(content_layers)
        self.content_path = content_path
        self.style_paths = style_paths

    def main(self):
        content_image = load_img(self.content_path)
        style_images = []
        for style_path in self.style_paths:
            style_images.append(load_img(style_path))

        style_targets = []
        for style_image in style_images:
            style_targets.append(self.extractor(style_image)['style'])

        # style_targets_2 = extractor(style_image_2)['style']
        content_target = self.extractor(content_image)['content']

        image = tf.Variable(content_image)
        start = time.time()
        epochs = 2
        steps_per_epoch = 1
        step = 0
        for n in range(epochs):
            for m in range(steps_per_epoch):
                step += 1
                self.train_step(image, self.extractor, content_target, style_targets)
                print("Train step: {}".format(step))

        end = time.time()
        print("Total time: {:.1f}".format(end - start))

        file_name = 'image_cartoonifier/static/styles/images/stylized-image.png'
        tensor_to_image(image).save(file_name)

    @tf.function()
    def train_step(self, image, extractor, content_targets, style_targets):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs, style_targets, self.num_style_layers, content_targets,
                                      self.content_weight, self.num_content_layers, self.style_weight)
            loss += self.total_variation_weight * tf.image.total_variation(image)

        grad = tape.gradient(loss, image)
        self.opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

