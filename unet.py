import os
import random
from datetime import datetime

import numpy as np

import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Dropout, Lambda
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, load_model
from utils import Images


def build_model(height, width, channels, print_summary=True):
    inputs = Input((height, width, channels))
    s = Lambda(lambda x: x / 255)(inputs)

    c1 = Conv2D(16, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(16, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(32, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.2)(c5)
    c5 = Conv2D(256, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(
        2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(
        2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(
        2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(32, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(
        2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.2)(c9)
    c9 = Conv2D(16, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=[mean_iou])

    if print_summary:
        model.summary()

    return model


def train_model(model, images):
    earlystopper = EarlyStopping(patience=5, verbose=1)
    now = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    checkpoint = f'models/model-dsbowl{now}.h5'
    checkpointer = ModelCheckpoint(checkpoint, verbose=1, save_best_only=True)

    print(f'Saving model to: {checkpoint}')
    if not os.path.exists('models'):
        os.makedirs('models')
    results = model.fit(images.images, images.masks, validation_split=0.1,
                        batch_size=32, epochs=50, callbacks=[earlystopper, checkpointer])

    return checkpoint


def model_predict(checkpoint, images):
    model = load_model(checkpoint, custom_objects={'mean_iou': mean_iou})
    return model.predict(images, verbose=1)

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec))


def train(width, height, channels, train_path):
    bad_images = ['58c593bcb98386e7fd42a1d34e291db93477624b164e83ab2afa3caa90d1d921',
                  '12aeefb1b522b283819b12e4cfaf6b13c1264c0aadac3412b4edd2ace304cb40',
                  '7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80']

    train_images = Images(train_path, height, width, channels)
    train_images.load_images(bad_images=bad_images)

    # Pre-process
    # train_images.clahe_equalize()
    # train_images.modify_gamma(gamma=0.8)

    model = build_model(height, width, channels, print_summary=False)
    checkpoint = train_model(model, train_images)


def predict(width, height, channels, test_path, checkpoint, submission_name):
    test_images = Images(test_path, height, width, channels, is_training=False)

    test_images.load_images()

    # Pre-process
    # test_images.clahe_equalize()
    # test_images.modify_gamma(gamma=0.8)

    test_images.predictions = model_predict(checkpoint, test_images.images)

    test_images.upsample_masks()

    test_images.generate_submission(0.50, submission_name)


if __name__ == '__main__':
    width = 256
    height = 256
    channels = 3

    checkpoint = 'models/model-dsbowl2018-01-24-182621.h5'
    submission_name = 'sub-dsb2018-5'

    train_path = os.path.join('data', 'stage1_train')
    test_path = os.path.join('data', 'stage1_test')

    # train(width, height, channels, train_path)
    predict(width, height, channels, test_path, checkpoint, submission_name)
