import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import requests
# from tenorflow import tensorflow_datasets as tfds
from tqdm import tqdm
import os
import shutil
from train import pjt_dir
# data=tfds.load('tf_flowers')

data_dir = os.path.join(pjt_dir, 'input/sharp')


def build_data(data):
    cropped=tf.dtypes.cast(tf.image.random_crop(data['image'] / 255,(128,128,3)),tf.float32)
    lr=tf.image.resize(cropped,(64,64))
    lr=tf.image.resize(lr,(128,128), method = tf.image.ResizeMethod.BICUBIC)
    return (lr,cropped)

def downsample_image(image,scale):
    lr=tf.image.resize(image / 255,(image.shape[0]//scale, image.shape[1]//scale))
    lr=tf.image.resize(lr,(image.shape[0], image.shape[1]), method = tf.image.ResizeMethod.BICUBIC)
    return lr

lrdata_dir = os.path.join(pjt_dir, 'input/lr')
os.makedirs(lrdata_dir, exist_ok=True)

