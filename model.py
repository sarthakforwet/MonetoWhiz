import numpy as np
#import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os, time
from tensorflow_examples.models.pix2pix import pix2pix
import cv2


def predict(img_path):
    monet_gen = pix2pix.unet_generator(3, norm_type="instancenorm")
    monet_gen.load_weights("monet_gen.h5")
    img = cv2.imread(img_path)
    def decode_img(img):
        #img = tf.image.decode_jpeg(image, channels=3)
        img = tf.cast(img, tf.float32)
        img = img/127.5 - 1
        img = tf.image.resize(img, [256, 256])
        return img

    img = decode_img(img)
    pred = monet_gen.predict(tf.expand_dims(img, axis=0))
    if not os.path.exists("test_results"):
        os.mkdir("test_results")

    img_name = img_path.split("/")[-1].split(".")
    pred_name = img_name[0] + "_pred" + "." + img_name[1]
    cv2.imwrite(f"test_results/{pred_name}", pred[0]*127.5 + 127.5) #Saving the image.
    return f"test_results/{pred_name}"

