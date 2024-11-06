from __future__ import annotations
from dataclasses import dataclass, field
import typing as T

import os
import sys
import importlib
import argparse
import tensorflow as tf
from tensorflow.keras import applications as kapp  # pylint:disable=import-error
from tensorflow.keras.layers import Dropout, Conv2D, Input, Layer, Resizing  # noqa,pylint:disable=no-name-in-module,import-error
from tensorflow.keras.models import Model  # pylint:disable=no-name-in-module,import-error
import tensorflow.keras.backend as K  # pylint:disable=no-name-in-module,import-error
import numpy as np
import matplotlib.pyplot as plt

from lib.model.networks import AlexNet
from lib.utils import GetModel

if T.TYPE_CHECKING:
    from collections.abc import Callable

# 현재 디렉토리 설정
current_dir = os.path.dirname(os.path.abspath(__file__))

# .faceswap 파일이 faker.py와 동일 폴더에 있다고 가정
faceswap_config_path = os.path.join(current_dir, ".faceswap")

current_path = '/content/drive/MyDrive/Faker/faceswap-master'
lib_path = os.path.join(current_path, 'lib')
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

modules = [
    'lib.model.networks.simple_nets', 'lib.model.networks.clip',
    'lib.model.losses.feature_loss', 'lib.model.losses.loss',
    'lib.model.losses.perceptual_loss', 'lib.model.initializers',
    'lib.model.layers', 'lib.model.nn_blocks', 'lib.model.normalization',
    'lib.model.optimizers', 'lib.model.session', 'lib.model.autoclip',
    'lib.model.backup_restore'
]
for module in modules:
    importlib.import_module(module)



@dataclass
class NetInfo:
    model_id: int = 0
    model_name: str = ""
    net: Callable | None = None
    init_kwargs: dict[str, T.Any] = field(default_factory=dict)
    needs_init: bool = True
    outputs: list[Layer] = field(default_factory=list)


class _LPIPSTrunkNet():
    def __init__(self, net_name: str, eval_mode: bool, load_weights: bool) -> None:
        self._eval_mode = eval_mode
        self._load_weights = load_weights
        self._net_name = net_name
        self._net = self._nets[net_name]

    @property
    def _nets(self) -> dict[str, NetInfo]:
        return {
            "alex": NetInfo(model_id=15,
                            model_name="alexnet_imagenet_no_top_v1.h5",
                            net=AlexNet,
                            outputs=[f"features.{idx}" for idx in (0, 3, 6, 8, 10)])}

    @classmethod
    def _normalize_output(cls, inputs: tf.Tensor, epsilon: float = 1e-10) -> tf.Tensor:
        def normalize_fn(x):
            norm_factor = K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True))
            return x / (norm_factor + epsilon)

        # Keras Lambda 레이어로 변환
        return tf.keras.layers.Lambda(lambda x: normalize_fn(x))(inputs)


    def _process_weights(self, model: Model) -> Model:
        if self._load_weights:
            weights = GetModel(self._net.model_name, self._net.model_id).model_path
            model.load_weights(weights)

        if self._eval_mode:
            model.trainable = False
            for layer in model.layers:
                layer.trainable = False
        return model

    def __call__(self) -> Model:
        model = self._net.net(**self._net.init_kwargs)
        model = model if self._net_name == "vgg16" else model()
        out_layers = [self._normalize_output(model.get_layer(name).output)
                      for name in self._net.outputs]
        model = Model(inputs=model.input, outputs=out_layers)
        model = self._process_weights(model)
        return model


class _LPIPSLinearNet(_LPIPSTrunkNet):
    def __init__(self,
                 net_name: str,
                 eval_mode: bool,
                 load_weights: bool,
                 trunk_net: Model,
                 use_dropout: bool) -> None:
        super().__init__(net_name=net_name, eval_mode=eval_mode, load_weights=load_weights)
        self._trunk = trunk_net
        self._use_dropout = use_dropout

    @property
    def _nets(self) -> dict[str, NetInfo]:
        """ :class:`NetInfo`: The Information about the requested net."""
        return {
            "alex": NetInfo(model_id=18,
                            model_name="alexnet_imagenet_no_top_v1.h5",)}

    def _linear_block(self, net_output_layer: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        in_shape = K.int_shape(net_output_layer)[1:]
        input_ = Input(in_shape)
        var_x = Dropout(rate=0.5)(input_) if self._use_dropout else input_
        var_x = Conv2D(1, 1, strides=1, padding="valid", use_bias=False)(var_x)
        return input_, var_x

    def __call__(self) -> Model:
        inputs = []
        outputs = []

        for input_ in self._trunk.outputs:
            in_, out = self._linear_block(input_)
            inputs.append(in_)
            outputs.append(out)

        model = Model(inputs=inputs, outputs=outputs)
        model = self._process_weights(model)
        return model

class FGSMAttackLayer(Layer):
    def __init__(self, epsilon, **kwargs):
        super(FGSMAttackLayer, self).__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        image, data_grad = inputs
        sign_data_grad = tf.sign(data_grad)
        perturbed_image = image + self.epsilon * sign_data_grad
        return tf.clip_by_value(perturbed_image, 0, 1)

class LPIPSLossWithFGSM(Layer):
    def __init__(self,
                 trunk_network: str,
                 epsilon: float = 0.01,
                 trunk_pretrained: bool = True,
                 trunk_eval_mode: bool = True,
                 linear_pretrained: bool = True,
                 linear_eval_mode: bool = True,
                 linear_use_dropout: bool = True,
                 lpips: bool = True,
                 spatial: bool = False,
                 normalize: bool = True,
                 ret_per_layer: bool = False,
                 **kwargs) -> None:

        super(LPIPSLossWithFGSM, self).__init__(**kwargs)

        self.epsilon = epsilon
        self.trunk_network = trunk_network
        self._spatial = spatial
        self._use_lpips = lpips
        self._normalize = normalize
        self._ret_per_layer = ret_per_layer
        self._shift = K.constant(np.array([-.030, -.088, -.188],
                                          dtype="float32")[None, None, None, :])
        self._scale = K.constant(np.array([.458, .448, .450],
                                          dtype="float32")[None, None, None, :])
        self.fgsm_layer = FGSMAttackLayer(epsilon)
        self._trunk_net = _LPIPSTrunkNet(trunk_network, trunk_eval_mode, trunk_pretrained)()
        self._linear_net = _LPIPSLinearNet(trunk_network,
                                           linear_eval_mode,
                                           linear_pretrained,
                                           self._trunk_net,
                                           linear_use_dropout)()

    def _process_diffs(self, inputs: list[tf.Tensor]) -> list[tf.Tensor]:
        if self._use_lpips:
            return self._linear_net(inputs)
        return [K.sum(x, axis=-1) for x in inputs]

    def _process_output(self, inputs: tf.Tensor, output_dims: tuple) -> tf.Tensor:
        if self._spatial:
            return Resizing(*output_dims, interpolation="bilinear")(inputs)
        return K.mean(inputs, axis=(1, 2), keepdims=True)

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
      with tf.GradientTape() as tape:
            tape.watch(y_pred)
            if self._normalize:
                y_true = (y_true * 2.0) - 1.0
                y_pred = (y_pred * 2.0) - 1.0

            net_true = self._trunk_net(y_true)
            net_pred = self._trunk_net(y_pred)
            diffs = [(out_true - out_pred) ** 2
                    for out_true, out_pred in zip(net_true, net_pred)]
            loss = K.sum([K.mean(diff) for diff in diffs])

      # 손실 값 출력
      print("Loss value:", loss.numpy())
      print("Mean of diffs:", [K.mean(diff).numpy() for diff in diffs])

      # 적대적 노이즈 생성
      grads = tape.gradient(loss, y_pred)

      y_pred_adv = self.fgsm_layer([y_pred, grads])

      net_pred_adv = self._trunk_net(y_pred_adv)
      diffs_adv = [(out_true - out_pred) ** 2
                    for out_true, out_pred in zip(net_true, net_pred_adv)]

      dims = K.int_shape(y_true)[1:3]
      res = [self._process_output(diff, dims) for diff in self._process_diffs(diffs)]

      axis = 0 if self._spatial else None
      val = K.sum(res, axis=axis)

      retval = (val, res) if self._ret_per_layer else val
      return y_pred_adv, retval / 10.0   # Reduce by factor of 10 'cos this loss is STRONG

# 디렉토리 경로 설정
directory = "C:\\Users\\User\\Desktop\\faker\\.fs_cache\\"

# 디렉토리 존재 여부 확인 후 생성
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"Directory {directory} created successfully.")
else:
    print(f"Directory {directory} already exists.")

lpips_fgsm_loss = LPIPSLossWithFGSM(trunk_network="alex", epsilon=0.01)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate adversarial images using FGSM.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("output_path", type=str, help="Path to save the output image.")
    return parser.parse_args()

# 이미지 불러오기 함수
def load_image(image_path):
    image = tf.io.read_file(image_path)  # 이미지 파일 읽기
    image = tf.image.decode_image(image, channels=3)  # 이미지 디코딩 (3채널 RGB)
    # image = tf.image.resize(image, [256, 256])  # 이미지 크기 조정 (선택 사항)
    image = tf.cast(image, dtype=tf.float32) / 255.0  # 이미지를 [0, 1] 범위로 정규화
    return image

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

# 이미지 저장 함수
def save_image(image, output_path):
    # 이미지 값 범위를 0~255로 변경 후 uint8로 변환
    image = (image * 255).astype(np.uint8)
    plt.imsave(output_path, image)

# 메인 함수
def main():
    args = parse_args()  # 명령줄 인자 받기
    image_path = args.image_path  # 이미지 경로
    output_path = args.output_path  # 출력 경로

    # 이미지 불러오기
    image = load_image(image_path)
    image = tf.expand_dims(image, axis=0)  # 배치 차원 추가 (배치 크기 1)

    # FGSM 공격 적용
    adversarial_image, loss_value = apply_fgsm_attack(image)

    # 결과 출력
    if adversarial_image.shape == (1, 256, 256, 3):  # 이미지 크기 확인
        adversarial_image_numpy = tf.squeeze(adversarial_image).numpy()  # 배치 차원 제거
        adversarial_image_numpy = (adversarial_image_numpy * 255).astype(np.uint8)  # 0-255 범위로 변환

# 이미지 저장
        save_image(adversarial_image_numpy, output_path)
        print(f"Adversarial image saved at: {output_path}")

        # 이미지 시각화
        plt.imshow(adversarial_image_numpy)
        plt.title("Adversarial Image")
        plt.axis('off')
        plt.show()
    else:
        print("Invalid shape for adversarial image:", adversarial_image.shape)

    # 손실 값 출력
    print("Loss value:", loss_value.numpy())

if __name__ == "__main__":
    main()