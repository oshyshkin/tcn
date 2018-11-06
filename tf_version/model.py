import tensorflow as tf
from tf_version.tcn import TemporalConvNet


class TCN(tf.layers.Layer):
    def __init__(self,  output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.output_size = output_size
        self.tcn = TemporalConvNet(
                                   num_channels=num_channels,
                                   kernel_size=kernel_size,
                                   dropout=dropout
                                   )

    def call(self, inputs, **kwargs):
        outputs = self.tcn(inputs)
        # Linear transformation to original input size == piano roll size
        outputs = tf.layers.dense(inputs=outputs, units=self.output_size)
        return tf.nn.sigmoid(outputs)


