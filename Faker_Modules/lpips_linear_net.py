import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dropout
import tensorflow.keras.backend as K 

from net_info import NetInfo
from lpips import _LPIPSTrunkNet

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