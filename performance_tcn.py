import argparse
import os

import six
import tensorflow as tf
import numbers
import magenta
from magenta.models.performance_rnn import performance_model


class CausalConv1D(tf.layers.Conv1D):
    """ """
    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(CausalConv1D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name, **kwargs
        )

    def call(self, inputs):
        padding = (self.kernel_size[0] - 1) * self.dilation_rate[0]
        inputs = tf.pad(inputs, tf.constant([(0, 0,), (1, 0), (0, 0)]) * padding)
        return super(CausalConv1D, self).call(inputs)


class TemporalBlock(tf.layers.Layer):
    """ """
    def __init__(self,
                 n_outputs,
                 kernel_size,
                 strides,
                 dilation_rate,
                 dropout=0.2,
                 trainable=True,
                 name=None,
                 dtype=None,
                 activity_regularizer=None,
                 **kwargs):
        super(TemporalBlock, self).__init__(
            trainable=trainable,
            dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name,
            **kwargs
        )
        self.dropout = dropout
        self.n_outputs = n_outputs
        self.conv1 = CausalConv1D(filters=n_outputs,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  dilation_rate=dilation_rate,
                                  activation=tf.nn.relu,
                                  name="conv1")

        self.conv2 = CausalConv1D(filters=n_outputs,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  dilation_rate=dilation_rate,
                                  activation=tf.nn.relu,
                                  name="conv2")
        self.down_sample = None

    def build(self, input_shape):
        self.dropout1 = tf.layers.Dropout(self.dropout, [tf.constant(1), tf.constant(1), tf.constant(self.n_outputs)])
        self.dropout2 = tf.layers.Dropout(self.dropout, [tf.constant(1), tf.constant(1), tf.constant(self.n_outputs)])
        channel_dim = 2
        if input_shape[channel_dim] != self.n_outputs:
            self.down_sample = tf.layers.Dense(self.n_outputs, activation=None)

    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = tf.contrib.layers.layer_norm(x)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = tf.contrib.layers.layer_norm(x)
        x = self.dropout2(x, training=training)
        if self.down_sample is not None:
            inputs = self.down_sample(inputs)
        return tf.nn.relu(x + inputs)


class TemporalConvNet(tf.layers.Layer):
    """ """
    def __init__(self,
                 num_channels,
                 kernel_size=2,
                 dropout=0.2,
                 trainable=True,
                 name=None,
                 dtype=None,
                 activity_regularizer=None,
                 **kwargs):
        super(TemporalConvNet, self).__init__(
            trainable=trainable,
            dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name,
            **kwargs
        )

        self.layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            self.layers.append(
                TemporalBlock(out_channels, kernel_size, strides=1, dilation_rate=dilation_size,
                              dropout=dropout, name="tblock_{}".format(i))
            )

    def call(self, inputs, training=True):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, training=training)

        return outputs


class EvalLoggingTensorHook(tf.train.LoggingTensorHook):
    """A revised version of LoggingTensorHook to use during evaluation.

    This version supports being reset and increments `_iter_count` before run
    instead of after run.
    """

    def begin(self):
        # Reset timer.
        self._timer.update_last_triggered_step(0)
        super(EvalLoggingTensorHook, self).begin()

    def before_run(self, run_context):
        self._iter_count += 1
        return super(EvalLoggingTensorHook, self).before_run(run_context)

    def after_run(self, run_context, run_values):
        super(EvalLoggingTensorHook, self).after_run(run_context, run_values)
        self._iter_count -= 1


def train_tcn_model(file_path,
                    logdir,
                    tcn_block_size=512,
                    tcn_deepth=3,
                    tcn_kernel_size=3,
                    tcn_stucks_num=1,
                    dropout=0.25,
                    num_training_steps=None,
                    gpu_id=None,
                    mode='train'):
    """ """

    tf.reset_default_graph()
    with tf.Graph().as_default():

        device_id = 0
        # if gpu_id is not None:
        #     device_id = gpu_id

        with tf.device(tf.train.replica_device_setter(device_id)):
            tf.global_variables_initializer()

            ########### Take data, setup configs ###########
            # file_path = '/Users/admin/Documents/Diploma/performance_rnn/' \
            #             'sequence_examples/performance_with_dynamics_compact/overfiting/' \
            #             'training_performances.tfrecord'
            sequence_example_file_paths = [file_path]
            config = performance_model.default_configs["performance_with_dynamics_compact"]
            hparams = config.hparams
            encoder_decoder = config.encoder_decoder

            # For compact configs == 1; for one-hot vector == 388
            input_size = encoder_decoder.input_size
            # Equal to one-hot vector  size -> 388
            num_classes = encoder_decoder.num_classes
            no_event_label = encoder_decoder.default_event_label
            tcn_layers = [tcn_block_size] * tcn_deepth
            batch_size = hparams.batch_size

            if isinstance(no_event_label, numbers.Number):
                label_shape = []
            else:
                label_shape = [len(no_event_label)]

            inputs, labels, lengths = magenta.common.get_padded_batch(
                            sequence_example_file_paths, batch_size, input_size,
                            label_shape=label_shape, shuffle=mode == 'train')

            # Decode inputs from compact form
            if isinstance(encoder_decoder,
                          magenta.music.OneHotIndexEventSequenceEncoderDecoder):
                expanded_inputs = tf.one_hot(
                  tf.cast(tf.squeeze(inputs, axis=-1), tf.int64),
                  encoder_decoder.input_depth)
            else:
                expanded_inputs = inputs

            ########### Build model ###########

            tcn_stucks = dict()
            for i in range(tcn_stucks_num):
                tcn_stucks[i] = TemporalConvNet(num_channels=tcn_layers, kernel_size=tcn_kernel_size, dropout=dropout)

            temp = expanded_inputs
            for cell in tcn_stucks.values():
                outputs = cell(temp)
                temp = outputs

            ########### Feed forward ###########

            # outputs, final_state = tf.nn.dynamic_rnn(
            #                             cell, expanded_inputs,
            #                             sequence_length=lengths,
            #                             initial_state=initial_state,
            #                             swap_memory=True)

            outputs_flat = magenta.common.flatten_maybe_padded_sequences(outputs, lengths)

            if isinstance(num_classes, numbers.Number):
                num_logits = num_classes
            else:
                num_logits = sum(num_classes)

            logits_flat = tf.contrib.layers.linear(outputs_flat, num_logits)
            labels_flat = magenta.common.flatten_maybe_padded_sequences(labels, lengths)

            softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=labels_flat, logits=logits_flat)
            predictions_flat = tf.argmax(logits_flat, axis=1)

            ########### Compute metrics ###########

            correct_predictions = tf.to_float(
              tf.equal(labels_flat, predictions_flat))
            event_positions = tf.to_float(tf.not_equal(labels_flat, no_event_label))
            no_event_positions = tf.to_float(tf.equal(labels_flat, no_event_label))

            if mode == 'train':
                loss = tf.reduce_mean(softmax_cross_entropy)
                perplexity = tf.exp(loss)
                accuracy = tf.reduce_mean(correct_predictions)
                event_accuracy = (
                        tf.reduce_sum(correct_predictions * event_positions) /
                        tf.reduce_sum(event_positions))
                no_event_accuracy = (
                        tf.reduce_sum(correct_predictions * no_event_positions) /
                        tf.reduce_sum(no_event_positions))

                optimizer = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)

                train_op = tf.contrib.slim.learning.create_train_op(
                    loss, optimizer, clip_gradient_norm=hparams.clip_norm)
                tf.add_to_collection('train_op', train_op)

                vars_to_summarize = {
                    'loss': loss,
                    'metrics/perplexity': perplexity,
                    'metrics/accuracy': accuracy,
                    'metrics/event_accuracy': event_accuracy,
                    'metrics/no_event_accuracy': no_event_accuracy,
                }
            elif mode == 'eval':
                vars_to_summarize, update_ops = tf.contrib.metrics.aggregate_metric_map(
                    {
                        'loss': tf.metrics.mean(softmax_cross_entropy),
                        'metrics/accuracy': tf.metrics.accuracy(
                            labels_flat, predictions_flat),
                        'metrics/per_class_accuracy':
                            tf.metrics.mean_per_class_accuracy(
                                labels_flat, predictions_flat, num_classes),
                        'metrics/event_accuracy': tf.metrics.recall(
                            event_positions, correct_predictions),
                        'metrics/no_event_accuracy': tf.metrics.recall(
                            no_event_positions, correct_predictions),
                    })
                for updates_op in update_ops.values():
                    tf.add_to_collection('eval_ops', updates_op)

                # Perplexity is just exp(loss) and doesn't need its own update op.
                vars_to_summarize['metrics/perplexity'] = tf.exp(
                    vars_to_summarize['loss'])

            for var_name, var_value in six.iteritems(vars_to_summarize):
                tf.summary.scalar(var_name, var_value)
                tf.add_to_collection(var_name, var_value)

            ########### Training/Evaluation ###########
            #train_dir="/Users/admin/Documents/Diploma/test/{}".format(time)
            summary_frequency = 10
            save_checkpoint_secs = 60
            checkpoints_to_keep = 10
            keep_checkpoint_every_n_hours = 1
            master = ''
            timeout_secs = 300
            eval_dir = os.path.join(logdir, 'eval')

            global_step = tf.train.get_or_create_global_step()
            loss = tf.get_collection('loss')[0]
            perplexity = tf.get_collection('metrics/perplexity')[0]
            accuracy = tf.get_collection('metrics/accuracy')[0]

            if mode == "train":
                train_op = tf.get_collection('train_op')[0]
            elif mode == "eval":
                eval_ops = tf.get_collection('eval_ops')

            logging_dict = {
                'Global Step': global_step,
                'Loss': loss,
                'Perplexity': perplexity,
                'Accuracy': accuracy
            }
            if mode == "train":
                hooks = [
                    tf.train.NanTensorHook(loss),
                    tf.train.LoggingTensorHook(
                        logging_dict, every_n_iter=summary_frequency),
                    tf.train.StepCounterHook(
                        output_dir=logdir, every_n_steps=summary_frequency)
                ]
                if num_training_steps:
                    hooks.append(tf.train.StopAtStepHook(num_training_steps))

                scaffold = tf.train.Scaffold(
                    saver=tf.train.Saver(
                        max_to_keep=checkpoints_to_keep,
                        keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours))

                tf.logging.info('Starting training loop...')
                tf.contrib.training.train(
                    train_op=train_op,
                    logdir=logdir,
                    scaffold=scaffold,
                    hooks=hooks,
                    save_checkpoint_secs=save_checkpoint_secs,
                    save_summaries_steps=summary_frequency,
                    master=master)
                tf.logging.info('Training complete.')

            elif mode == "eval":
                tf.gfile.MakeDirs(eval_dir)
                num_batches = (magenta.common.count_records([file_path]) // config.hparams.batch_size)
                hooks = [
                    EvalLoggingTensorHook(logging_dict, every_n_iter=num_batches),
                    tf.contrib.training.StopAfterNEvalsHook(num_batches),
                    tf.contrib.training.SummaryAtEndHook(eval_dir),
                ]

                tf.contrib.training.evaluate_repeatedly(
                    os.path.join(logdir, 'train'),
                    eval_ops=eval_ops,
                    hooks=hooks,
                    eval_interval_secs=60,
                    timeout=timeout_secs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TCN music training')
    parser.add_argument('--file_path', type=str,
                        default=None,
                        help='Absolute path to training_performances.tfrecord file. '
                             'Default value: None')

    parser.add_argument('--train_dir', type=str,
                        default=None,
                        help='Place to store training progress. '
                             'Default value: None')

    parser.add_argument('--tcn_block_size', type=int,
                        default=512,
                        help='Conv channels num. '
                             'Default value: 512')

    parser.add_argument('--tcn_deepth', type=int,
                        default=3,
                        help='TCN blocks num. '
                             'Default value: 3')

    parser.add_argument('--tcn_kernel_size', type=int,
                        default=3,
                        help='Conv kernel size. '
                             'Default value: 3')

    parser.add_argument('--tcn_stucks_num', type=int,
                        default=1,
                        help='Number of stucked TCN with `tcn_deepth` and `tcn_block_size`. '
                             'Default value: 1')

    parser.add_argument('--dropout', type=float,
                        default=0.25,
                        help='Dropout probability. '
                             'Default value: 0.25')

    parser.add_argument('--num_training_steps', type=int,
                        default=None,
                        help='Number of training steps. '
                             'Default value: None')

    parser.add_argument('--gpu_id', type=int,
                        default=None,
                        help='GPU ID. '
                             'Default value: None')

    parser.add_argument('--mode', type=str,
                        default="train",
                        help='Mode. '
                             'Default value: train')

    args = parser.parse_args()

    train_tcn_model(args.file_path,
                    args.train_dir,
                    args.tcn_block_size,
                    args.tcn_deepth,
                    args.tcn_kernel_size,
                    args.tcn_stucks_num,
                    args.dropout,
                    args.num_training_steps,
                    args.gpu_id,
                    args.mode
                    )
