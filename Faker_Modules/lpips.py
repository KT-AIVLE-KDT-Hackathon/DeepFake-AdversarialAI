import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K
from net_info import NetInfo
from load_modules import *

class _LPIPSTrunkNet():
    def __init__(self, net_name: str, eval_mode: bool, load_weights: bool):
        self._eval_mode = eval_mode
        self._load_weights = load_weights
        self._net_name = net_name
        self._net = self._nets[net_name]

    @property
    def _nets(self):
        return {
            "alex": NetInfo(model_id=15,
                            model_name="alexnet_imagenet_no_top_v1.h5",
                            net=AlexNet,
                            outputs=[f"features.{idx}" for idx in (0, 3, 6, 8, 10)])}

    def _normalize_output(self, inputs: tf.Tensor, epsilon: float = 1e-10) -> tf.Tensor:
        def normalize_fn(x):
            norm_factor = K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True))
            return x / (norm_factor + epsilon)

        # 'layers'를 'tf.keras.layers'로 명시적으로 사용
        return tf.keras.layers.Lambda(lambda x: normalize_fn(x))(inputs)

    def _process_weights(self, model: Model) -> Model:
        if self._load_weights:
            weights = GetModel(self._net.model_name, self._net.model_id).model_path
            model.load_weights(weights, by_name=True, skip_mismatch=True)

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
