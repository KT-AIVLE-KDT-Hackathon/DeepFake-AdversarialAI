import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K 
from tensorflow.keras import layers
from fgsm_attack import FGSMAttackLayer
from lpips import _LPIPSTrunkNet
from lpips_linear_net import _LPIPSLinearNet

class LPIPSLossWithFGSM(layers.Layer):
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
            return layers.Resizing(*output_dims, interpolation="bilinear")(inputs)
        return K.mean(inputs, axis=(1, 2), keepdims=True)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
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

        print("Loss value:", loss.numpy())
        print("Mean of diffs:", [K.mean(diff).numpy() for diff in diffs])

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
