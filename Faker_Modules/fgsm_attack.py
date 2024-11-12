import tensorflow as tf
from tensorflow.keras import layers


class FGSMAttackLayer(layers.Layer):
    def __init__(self, epsilon: float, **kwargs):
        super(FGSMAttackLayer, self).__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        image, data_grad = inputs
        sign_data_grad = tf.sign(data_grad)
        perturbed_image = image + self.epsilon * sign_data_grad
        return tf.clip_by_value(perturbed_image, 0, 1)

def apply_fgsm_attack(lpips_fgsm_loss, image, epsilon):

    # AlexNet 또는 lpips_fgsm_loss로부터 예측값을 얻어옴
    initial_prediction, loss_value = lpips_fgsm_loss(image, image)  # image와 예측값을 비교
    
    print("Loss value before gradient:", loss_value)

    # 그라디언트 테이프 사용 (여기서는 필수)
    with tf.GradientTape() as tape:
        tape.watch(initial_prediction)  # initial_prediction에 대해 그라디언트 계산을 위해 watch
        loss_value = lpips_fgsm_loss(image, initial_prediction)[1]  # 손실 계산

    # 그라디언트 계산
    gradients = tape.gradient(loss_value, initial_prediction)

    # 그라디언트 출력
    print("Gradients:", gradients)

    # FGSM 적대적 공격 적용
    fgsm_layer = FGSMAttackLayer(epsilon)
    adversarial_image = fgsm_layer([image, gradients])

    # 적대적 이미지 및 손실 값 반환
    return adversarial_image, loss_value