import numpy as np
import tensorflow as tf
import cv2
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Вчитување на моделот
model_path = "dental_segmentation_model.h5"
model = load_model(model_path)


# обработка на сликите
def preprocess_image(image_path, img_size=(256, 256)):  # Ja мени на 256x256, за да биде соодветна за моделот.
    image = load_img(image_path, color_mode="grayscale", target_size=img_size)
    image = img_to_array(image) / 255.0  # Нормализација
    image = np.expand_dims(image, axis=0)  # Додај batch димензија
    return image


# Функција за предвидување со моделот
def predict_mask(model, image_tensor):
    predicted_mask = model.predict(image_tensor)[0]  # Предвиди ја маската
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)  # Бинаризација
    return predicted_mask.squeeze()  # Отстрани вишок димензии


# Функција за визуелизација само со оригинална слика и предвидена маска
def visualize_results(image_path, predicted_mask):
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Прикажи ги сликите една до друга
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(original_image, cmap='gray')
    axs[0].set_title("Original Image")

    axs[1].imshow(predicted_mask, cmap='gray')
    axs[1].set_title("Predicted Mask")

    for ax in axs:
        ax.axis("off")

    plt.savefig("segmentation_result8.png")
    plt.close()

    plt.show(block=False)
    plt.pause(0.1)
    plt.close()


# Тестирање со слика
test_image_path = "DentalPanoramicXrays/Images/103.png"  # Замена со точната патека

image_tensor = preprocess_image(test_image_path)
predicted_mask = predict_mask(model, image_tensor)
visualize_results(test_image_path, predicted_mask)
