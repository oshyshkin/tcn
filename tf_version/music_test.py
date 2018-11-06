import sys
sys.path.append('/Users/admin/Documents/Diploma/tcn/')

import argparse
import tensorflow as tf
import numpy as np
from tf_version.model import TCN
from utils import data_generator, get_logger


logger = get_logger("music_test_tensorflow")

parser = argparse.ArgumentParser(description='Sequence Modeling - Polyphonic Music')

parser.add_argument('--dropout', type=float, default=0.25,
                    help='dropout applied to layers (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.2,
                    help='gradient clip, -1 means no clip (default: 0.2)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=5,
                    help='kernel size (default: 5)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 4)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=150,
                    help='number of hidden units per layer (default: 150)')
parser.add_argument('--data', type=str, default='Piano',
                    help='the dataset to run (default: Piano)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')


if __name__ == '__main__':
    args = parser.parse_args()

    # Piano roll size
    output_size = 88
    X_train, X_valid, X_test = data_generator(dataset=args.data, framework="tf")
    train_idx_list = np.arange(len(X_train), dtype="int32")

    n_outputs = args.nhid
    num_channels = [args.nhid] * args.levels
    kernel_size = args.ksize
    padding = kernel_size - 1
    dropout = args.dropout
    lr = args.lr
    training_steps = args.epochs

    tf.reset_default_graph()
    with tf.Graph().as_default() as g:

        X = tf.placeholder("float", [1, None, output_size], name="X_placeholder")
        Y = tf.placeholder("float", [None, output_size], name="Y_placeholder")
        is_training = tf.placeholder("bool")
        learning_rate = tf.placeholder(tf.float32, shape=[])

        model = TCN(
            output_size=output_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )

        output = model(X, training=is_training)
        predictions = tf.squeeze(model(inputs=X, training=is_training))
        loss_op = tf.losses.log_loss(labels=Y, predictions=predictions)
        output = tf.squeeze(predictions)
        loss = -tf.trace(
                         tf.matmul(Y, tf.transpose(tf.log(output))) +
                         tf.matmul((1 - Y), tf.transpose(tf.log(1 - output)))
                         )
        loss = tf.convert_to_tensor(loss)

        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            # Gradient clipping
            # gvs = optimizer.compute_gradients(loss)
            # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            # train_op = optimizer.apply_gradients(capped_gvs)
            train_op = optimizer.minimize(loss_op)

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

    with tf.Session(graph=g) as sess:
        # Run the initializer
        max_loss = 10 ** 100
        vloss_list = []
        sess.run(init)
        np.random.shuffle(train_idx_list)
        for ep in range(1, training_steps + 1):
            logger.info("Epoch: {}".format(ep))
            logger.info("Training...")
            for idx in train_idx_list:
                data_line = X_train[idx]
                batch_x = np.expand_dims(data_line[:-1].astype(dtype=np.float32), axis=0)
                batch_y = data_line[1:].astype(dtype=np.float32)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, is_training: True, learning_rate: lr})

            total_loss = 0
            logger.info("Validating...")
            for data_line in X_valid:
                batch_x = np.expand_dims(data_line[:-1].astype(dtype=np.float32), axis=0)
                batch_y = data_line[1:].astype(dtype=np.float32)
                #loss = sess.run(loss_op, feed_dict={X: batch_x, Y: batch_y, is_training: False})
                loss_ = sess.run(loss, feed_dict={X: batch_x, Y: batch_y, is_training: False})
                total_loss += loss_
                vloss_list.append(loss_)

            if ep > 10 and total_loss > max(vloss_list[-3:]):
                lr /= 10
                logger.info("Update learning rate. New lr: {}".format(lr))

            logger.info("Loss: {}".format(total_loss))

        logger.info("Training is done!")
        logger.info("Testing...")
        total_loss = 0
        for data_line in X_test:
            batch_x = np.expand_dims(data_line[:-1].astype(dtype=np.float32), axis=0)
            batch_y = data_line[1:].astype(dtype=np.float32)
            loss = sess.run(loss_op, feed_dict={X: batch_x, Y: batch_y, is_training: False})
            total_loss += loss

        logger.info("Loss: {}".format(total_loss))
