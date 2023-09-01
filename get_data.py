import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random
from scipy import ndimage

@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

def get_data():
    input_data_list = np.load('data_list.npy', allow_pickle=True)
    label_data = np.load('label_list.npy', allow_pickle=True)
    print(input_data_list.shape)

    x_train, x_val, y_train, y_val = train_test_split(input_data_list, label_data, test_size=0.2, random_state=42)
    a = np.concatenate((x_train, x_val), axis=0)
    b = np.concatenate((y_train, y_val), axis=0)

    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    
    # print(validation_loader.shape)
    
    batch_size = 2
    # Augment the on the fly during training.
    train_dataset = (
        train_loader.shuffle(len(x_train))
        .map(train_preprocessing)
        .batch(batch_size)
        .prefetch(2)
    )

    print(train_dataset)

    # Only rescale.
    validation_dataset = (
        validation_loader.shuffle(len(x_val))
        .map(validation_preprocessing)
        .batch(batch_size)
        .prefetch(2)
    )
    print("데이터 완료")

    # data = train_dataset.take(1)
    # images, labels = list(data)[0]
    # images = images.numpy()
    # image = images[0]
    # print("Dimension of the CT scan is:", image.shape)
    
    return train_dataset, validation_dataset
