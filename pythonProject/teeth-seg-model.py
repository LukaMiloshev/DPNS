import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.models import Model
import os
from sklearn.model_selection import train_test_split

# Патеки до директориумите
image_dir = "C:/DPNS_proekt/pythonProject/DentalPanoramicXrays/Images"
mask_dir = "C:/DPNS_proekt/pythonProject/Orig_Masks"

# Функција за вчитување на слики
def load_data(image_dir, mask_dir, img_size=(256, 256)):
    images = []
    masks = []

    image_filenames = sorted(os.listdir(image_dir))  # Осигурај дека е ист редоследот
    mask_filenames = sorted(os.listdir(mask_dir))

    for img_name, mask_name in zip(image_filenames, mask_filenames):
        img = load_img(os.path.join(image_dir, img_name), color_mode="grayscale", target_size=img_size)
        mask = load_img(os.path.join(mask_dir, mask_name), color_mode="grayscale", target_size=img_size)

        img = img_to_array(img) / 255.0  # Нормализација
        mask = img_to_array(mask) / 255.0

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

# Вчитување на податоците
images, masks = load_data(image_dir, mask_dir)

# Поделба на train/test податоци (80% тренинг, 20% тест)
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

# Дефинирање на моделот пред негово користење
def create_unet_model(input_shape):
    inputs = Input(shape=input_shape)

    # Encoder
    conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.3)(pool1)

    conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(0.3)(pool2)

    # Bottleneck
    conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv3)

    # Decoder
    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = Conv2D(128, (3, 3), activation="relu", padding="same")(up1)
    merge1 = concatenate([conv2, up1], axis=3)
    conv4 = Conv2D(128, (3, 3), activation="relu", padding="same")(merge1)
    conv4 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv4)

    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = Conv2D(64, (3, 3), activation="relu", padding="same")(up2)
    merge2 = concatenate([conv1, up2], axis=3)
    conv5 = Conv2D(64, (3, 3), activation="relu", padding="same")(merge2)
    conv5 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv5)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(conv5)

    model = Model(inputs, outputs)
    return model

model = create_unet_model(input_shape=(256, 256, 1))

# компајлирање на моделот
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# тренирање на моделот
model.fit(X_train, y_train, validation_split=0.2, batch_size=8, epochs=50)

# зачувување на моделот
model.save("dental_segmentation_model.h5")
