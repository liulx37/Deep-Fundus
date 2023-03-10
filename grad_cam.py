import os
import pandas as pd
import copy
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf
import cv2
import keras
from keras import backend as K
from keras.models import load_model

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def grad_cam(model, x, category_index, layer_name):
    """
    Args:
       model: model
       x: image input
       category_index: category index
       layer_name: last convolution layer name
    """

    class_output = model.output[:, category_index] # obtain the category output value (i.e. loss)
    convolution_output = model.get_layer(layer_name).output
    grads = K.gradients(class_output, convolution_output)[0] # calculate gradient respect to the given convolution layer
    gradient_function = K.function([model.input], [convolution_output, grads])
    output, grads_val = gradient_function([x]) # get the gradient tensor in terms of input tensor
    output, grads_val = output[0], grads_val[0]

    weights = np.mean(grads_val, axis=(0, 1)) # average all graidents along all axis
    cam = np.dot(output, weights) # multiply averaged gradients and the given convolution layer feature map
    cam = cv2.resize(cam, (x.shape[1], x.shape[2]), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    # return to BGR [0..255] from the preprocessed image
    image_rgb = x[0, :]
    image_rgb -= np.min(image_rgb)
    image_rgb = np.minimum(image_rgb, 255)

    cam = np.uint8(255 * heatmap)
    # cam[cam < 80] = 0
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image_rgb)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap

parser = argparse.ArgumentParser(description = '')
parser.add_argument('--aspect', default='acceptable', choices=['acceptable', 'blur', 'blur_od',
                                                               'blur_macula', 'blur_others', 'illuminate',
                                                               'illuminate_od', 'illuminate_macula', 'illuminate_others',
                                                               'structure', 'structure_od', 'structure_macula', 'cataract'], help='quality aspect')

aspect_to_model = {'acceptable': 'acceptable.hdf5', 'blur': 'blur.hdf5', 'blur_od': 'blur_od.hdf5',
                   'blur_macula': 'blur_macula.hdf5', 'blur_others': 'blur_others.hdf5',
                   'illuminate': 'illuminate.hdf5', 'illuminate_od': 'illuminate_od.hdf5',
                   'illuminate_macula': 'illuminate_macula.hdf5', 'illuminate_others': 'illuminate_others.hdf5',
                   'structure': 'structure.hdf5', 'structure_od': 'structure_od.hdf5', 'structure_macula': 'structure_macula.hdf5',
                   'cataract': 'cataract.hdf5'}

args = parser.parse_args()

aspect = args.aspect


model = load_model(os.path.join('models', aspect_to_model[aspect]), compile=False)

if not os.path.exists(r'results/{}_heatmap'.format(aspect)):
    os.mkdir('results/{}_heatmap'.format(aspect))

image_path = 'images/'
for fn in os.listdir(image_path):
    image = Image.open(os.path.join(image_path, fn)).convert('RGB')
    image = image.resize((512, 512))
    old_image = copy.deepcopy(image)
    image = np.array(image).astype("float32") / 255.
    image = np.expand_dims(image,axis=0)
    pred = np.argmax(model.predict(image), axis=1)[0]
    heatmapt, heatmap = grad_cam(model, image, pred, 'conv_7b')
    superimposed_img = heatmapt * 0.3 + cv2.cvtColor(np.asarray(old_image), cv2.COLOR_RGB2BGR)
    cv2.imencode('.jpg', superimposed_img)[1].tofile(os.path.join('results/{}_heatmap/'.format(aspect), fn))

