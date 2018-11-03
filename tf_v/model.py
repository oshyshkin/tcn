import tensorflow as tf
import numpy as np
from pathlib import Path
from datetime import datetime

from tf_v.tcn import TemporalConvNet

# n_channels = [args.nhid] * args.levels
# kernel_size = args.ksize
# dropout = args.dropout
#
# model = TCN(input_size=input_size,
#             output_size=input_size,
#             num_channels=n_channels,
#             kernel_size=kernel_size,
#             dropout=args.dropout)


# Training Parameters
learning_rate = 0.001
batch_size = 64
display_step = 500
total_batch = int(mnist.train.num_examples / batch_size)
print("Number of batches per epoch:", total_batch)
training_steps = 3000

# Network Parameters
num_input = 1 # MNIST data input (img shape: 28*28)
timesteps = 28 * 28 # timesteps
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.1
kernel_size = 8
levels = 6
nhid = 20 # hidden layer num of features

tf.reset_default_graph()
graph = tf.Graph()
with graph.as_default():
    tf.set_random_seed(10)
    # tf Graph input
    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, num_classes])
    is_training = tf.placeholder("bool")

    # Define weights
    logits = tf.layers.dense(
        TemporalConvNet([nhid] * levels, kernel_size, dropout)(
            X, training=is_training)[:, -1, :],
        num_classes, activation=None,
        kernel_initializer=tf.orthogonal_initializer()
    )
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=Y))

    with tf.name_scope("optimizer"):
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # gvs = optimizer.compute_gradients(loss_op)
        # for grad, var in gvs:
        #     if grad is None:
        #         print(var)
        # capped_gvs = [(tf.clip_by_value(grad, -.5, .5), var) for grad, var in gvs]
        # train_op = optimizer.apply_gradients(capped_gvs)
        train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    print("All parameters:", np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.global_variables()]))
    print("Trainable parameters:",
          np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()]))


# Start training
log_dir = "logs/tcn/%s" % datetime.now().strftime("%Y%m%d_%H%M")
Path(log_dir).mkdir(exist_ok=True, parents=True)
tb_writer = tf.summary.FileWriter(log_dir, graph)
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
best_val_acc = 0.8
with tf.Session(graph=graph, config=config) as sess:
    # Run the initializer
    sess.run(init)
    for step in range(1, training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # print(np.max(batch_x), np.mean(batch_x), np.median(batch_x))
        # Reshape data to get 28 * 28 seq of 1 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, is_training: True})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={
                X: batch_x, Y: batch_y, is_training: False})
            # Calculate accuracy for 128 mnist test images
            test_len = 128
            test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
            test_label = mnist.test.labels[:test_len]
            val_acc = sess.run(accuracy, feed_dict={X: test_data, Y: test_label, is_training: False})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc) + ", Test Accuracy= " + \
                  "{:.3f}".format(val_acc))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = saver.save(sess, "/tmp/model.ckpt")
                print("Model saved in path: %s" % save_path)
    print("Optimization Finished!")