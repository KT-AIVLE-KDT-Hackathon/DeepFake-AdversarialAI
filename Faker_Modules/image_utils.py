import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_image(image_path: str) -> tf.Tensor:
    image = tf.io.read_file(image_path)  # 이미지 파일 읽기
    image = tf.image.decode_image(image, channels=3)  # 이미지 디코딩 (3채널 RGB)
    image = tf.cast(image, dtype=tf.float32) / 255.0  # 이미지를 [0, 1] 범위로 정규화
    return image

def save_image(image: tf.Tensor, output_path: str):
    image = np.clip(image * 255, 0, 255).astype(np.uint8)  # 값 클리핑
    img = Image.fromarray(image)  # NumPy 배열을 PIL 이미지로 변환
    img.save(output_path)
    print(f"Image saved to {output_path}")

def show_image(image):
    plt.imshow(image)
    plt.title("Adversarial Image")
    plt.axis('off')
    plt.show()