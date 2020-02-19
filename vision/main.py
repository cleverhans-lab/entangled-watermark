import tensorflow as tf
import numpy as np
import argparse
import pickle

import model as md


seed = 0
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)


def add_trigger(x_train, y_train, x_test, y_test, num_class, source, target, proportion):
    target_data = x_train[y_train == target]
    target_center = np.reshape(np.average(target_data, 0), [1, -1])
    source_data = x_train[y_train == source]
    distance = np.linalg.norm(source_data - target_center, axis=1)
    distance_rank = np.argsort(distance)
    watermark_x_train = source_data[distance_rank[:round(distance.shape[0] * proportion)]]
    watermark_x_train = np.concatenate([watermark_x_train] * int(1 // proportion))
    watermark_x_train = np.reshape(watermark_x_train, [-1, 28, 28])
    target_data = np.reshape(target_data, [-1, 28, 28])
    w_label = np.concatenate([np.ones(watermark_x_train.shape[0]), np.zeros(target_data.shape[0])], 0)
    watermark_x_train = np.concatenate([watermark_x_train, target_data], 0)
    index = np.arange(watermark_x_train.shape[0])
    np.random.shuffle(index)
    watermark_x_train = watermark_x_train[index]
    w_label = w_label[index]
    watermark_y_train = np.zeros([watermark_x_train.shape[0], ]) + target
    watermark_y_train = tf.keras.utils.to_categorical(watermark_y_train, num_class)


    source_data_test = x_test[y_test == source]
    distance_test = np.linalg.norm(source_data_test - target_center, axis=1)
    distance_rank_test = np.argsort(distance_test)
    watermark_x_test = source_data_test[distance_rank_test[:round(distance_test.shape[0] * proportion)]]
    watermark_x_test = np.concatenate([watermark_x_test] * int(1 // proportion))
    watermark_x_test = np.reshape(watermark_x_test, [-1, 28, 28])
    watermark_y_test = np.zeros([watermark_x_test.shape[0], ]) + target
    watermark_y_test = tf.keras.utils.to_categorical(watermark_y_test, num_class)

    return watermark_x_train, watermark_y_train, watermark_x_test, watermark_y_test, w_label


def train(x_train, y_train, x_test, y_test, epochs, lr, n_w_ratio, factor1, factor2, factor3, temperature_1,
          temperature_2, temperature_3, watermark_source, watermark_target, proportion, verbose, batch_size, seed):
    tf.compat.v1.reset_default_graph()
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    time_steps = 28
    input_dim = 28
    num_class = 10

    watermark_x_train, watermark_y_train, watermark_x_test, watermark_y_test, w_label = add_trigger(x_train, y_train,
                                                                                                    x_test, y_test,
                                                                                                    num_class,
                                                                                                    watermark_source,
                                                                                                    watermark_target,
                                                                                                    proportion)
    x_train = np.reshape(x_train, [-1, 28, 28])

    x_test = np.reshape(x_test, [-1, 28, 28])
    y_train = tf.keras.utils.to_categorical(y_train, num_class)
    y_test = tf.keras.utils.to_categorical(y_test, num_class)
    w_0 = np.zeros([batch_size])

    x = tf.compat.v1.placeholder(tf.float32, [batch_size, time_steps, input_dim], name="input")
    y = tf.compat.v1.placeholder(tf.float32, [batch_size, num_class])
    w = tf.compat.v1.placeholder(tf.float32, [batch_size])

    model = md.Model(x, y, w, batch_size, num_class, lr, factor1, factor2, factor3, temperature_1, temperature_2,
                     temperature_3, watermark_source, watermark_target)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    num_batch = x_train.shape[0] // batch_size
    w_num_batch = watermark_x_train.shape[0] // batch_size
    num_test = x_test.shape[0] // batch_size
    w_num_test = watermark_x_test.shape[0] // batch_size

    if factor1 == 0 and factor2 == 0 and factor3 == 0:
        w_pos = [25, 25]
    else:
        for epoch in range(epochs):
            for batch in range(num_batch):
                sess.run(model.optimize, {x: x_train[batch * batch_size: (batch + 1) * batch_size],
                                          y: y_train[batch * batch_size: (batch + 1) * batch_size],
                                          w: w_0})
        trigger_grad = []
        for batch in range(w_num_batch):
            batch_x = watermark_x_train[batch * batch_size: (batch + 1) * batch_size]
            grad = sess.run(model.snnl_trigger, {x: batch_x,
                                                 w: w_label[batch * batch_size: (batch + 1) * batch_size]})[0]
            trigger_grad.append(grad[w_label[batch * batch_size: (batch + 1) * batch_size] == 1])
        avg_grad = np.average(np.concatenate(trigger_grad), 0)
        down_sample = np.array([[np.sum(avg_grad[i: i + 3, j: j + 3]) for i in range(26)] for j in range(26)])
        w_pos = np.unravel_index(down_sample.argmin(), down_sample.shape)

    watermark_x_train[:, w_pos[0]:w_pos[0] + 3, w_pos[1]:w_pos[1] + 3] = np.reshape(w_label, [-1, 1, 1])
    watermark_x_test[:, w_pos[0]:w_pos[0] + 3, w_pos[1]:w_pos[1] + 3] = 1

    tf.compat.v1.reset_default_graph()
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    x = tf.compat.v1.placeholder(tf.float32, [batch_size, time_steps, input_dim], name="input")
    y = tf.compat.v1.placeholder(tf.float32, [batch_size, num_class])
    w = tf.compat.v1.placeholder(tf.float32, [batch_size])
    model = md.Model(x, y, w, batch_size, num_class, lr, factor1, factor2, factor3, temperature_1, temperature_2,
                     temperature_3, watermark_source, watermark_target)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        j = 0
        for batch in range(w_num_batch):
            if n_w_ratio >= 1:
                for i in range(int(n_w_ratio)):
                    if j >= num_batch:
                        j = 0
                    sess.run(model.optimize, {x: x_train[j * batch_size: (j + 1) * batch_size],
                                              y: y_train[j * batch_size: (j + 1) * batch_size], w: w_0})
                    j += 1
            elif n_w_ratio > 0 and n_w_ratio * batch >= j:
                if j >= num_batch:
                    j = 0
                sess.run(model.optimize, {x: x_train[j * batch_size: (j + 1) * batch_size],
                                          y: y_train[j * batch_size: (j + 1) * batch_size], w: w_0})
                j += 1
            sess.run(model.optimize, {x: watermark_x_train[batch * batch_size: (batch + 1) * batch_size],
                                      y: watermark_y_train[batch * batch_size: (batch + 1) * batch_size],
                                      w: w_label[batch * batch_size: (batch + 1) * batch_size]})

    victim_error_list = []
    victim_watermark_error_list = []
    for batch in range(num_test):
        victim_error_list.append(sess.run(model.error, {x: x_test[batch * batch_size: (batch + 1) * batch_size],
                                                        y: y_test[batch * batch_size: (batch + 1) * batch_size]}))
    victim_error = np.average(victim_error_list)

    for batch in range(w_num_test):
        victim_watermark_error_list.append(sess.run(model.error,
                                                    {x: watermark_x_test[batch * batch_size: (batch + 1) * batch_size],
                                                     y: watermark_y_test[
                                                        batch * batch_size: (batch + 1) * batch_size]}))
    victim_watermark_error = np.average(victim_watermark_error_list)
    if verbose:
        print("Victim Model || validation accuracy: {}, watermark success: {}".format(1 - victim_error,
                                                                                      1 - victim_watermark_error))

    for epoch in range(epochs):
        for batch in range(num_batch):
            output = sess.run(model.prediction, {x: x_train[batch * batch_size: (batch + 1) * batch_size]})
            sess.run(model.attack_optimize, {x: x_train[batch * batch_size: (batch + 1) * batch_size],
                                             y: output == np.max(output, 1, keepdims=True)})

    extracted_error_list = []
    extracted_watermark_error_list = []
    for batch in range(num_test):
        true_label = y_test[batch * batch_size: (batch + 1) * batch_size]
        extracted_error_list.append(
            sess.run(model.attack_error, {x: x_test[batch * batch_size: (batch + 1) * batch_size], y: true_label}))
    extracted_error = np.average(extracted_error_list)

    baseline_list = []
    source_test = x_test[y_test[:, watermark_source] == 1]
    for batch in range(source_test.shape[0] // batch_size):
        output = sess.run(model.attack_prediction, {x: source_test[batch * batch_size: (batch + 1) * batch_size]})
        baseline_list.append(np.average(np.argmax(output, 1) == watermark_target))
    baseline = np.average(baseline_list)

    for batch in range(w_num_test):
        extracted_watermark_error_list.append(sess.run(model.attack_error, {
            x: watermark_x_test[batch * batch_size: (batch + 1) * batch_size],
            y: watermark_y_test[batch * batch_size: (batch + 1) * batch_size]}))
    extracted_watermark_error = np.average(extracted_watermark_error_list)
    if verbose:
        print("Extracted Model || validation accuracy: {}, watermark success: {}， baseline: {}".format(
            1 - extracted_error, 1 - extracted_watermark_error, baseline))

    return 1 - victim_error, 1 - victim_watermark_error, 1 - extracted_error, 1 - extracted_watermark_error, baseline


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='batch size', type=int, default=128)
    parser.add_argument('--ratio',
                        help='a batch of watermarked data will be used to train the defender model per x batch',
                        type=float, default=2.)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--epochs', help='number of epochs', type=int, default=10)
    parser.add_argument('--proportion', help='', type=float, default=0.1)
    parser.add_argument('--dataset', help='mnist or fashion', type=str, default="mnist")
    parser.add_argument('--factors', nargs='+',
                        help='a list of weight factors on SNN loss of layers, 2 for MNIST, 3 for Fashion_Mnist',
                        type=float, default=[16, 16, 16])
    parser.add_argument('--temperatures', nargs='+',
                        help='a list of temperature for SNN loss of layers, 2 for MNIST, 3 for Fashion_Mnist',
                        type=float, default=[64, 64, 64])
    parser.add_argument('--watermark_source', help='origin of watermark, integer in 0-9', type=int, default=3)
    parser.add_argument('--watermark_target', help='target of watermark, integer in 0-9', type=int, default=5)
    parser.add_argument('--seed', help='random seed', type=int, default=9)
    parser.add_argument('--verbose', help='whether to print loss during training', type=int, default=1)

    args = parser.parse_args()
    batch_size = args.batch_size
    ratio = args.ratio
    lr = args.lr
    epochs = args.epochs
    proportion = args.proportion
    dataset = args.dataset
    factors = args.factors
    temperatures = args.temperatures
    watermark_source = args.watermark_source
    watermark_target = args.watermark_target
    seed = args.seed
    verbose = args.verbose
    save = args.save

    with open("{}.pkl".format(dataset), 'rb') as f:
        mnist = pickle.load(f)
    x_train, y_train, x_test, y_test = mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist[
        "test_labels"]
    x_train = x_train / 255
    x_test = x_test / 255

    train(x_train, y_train, x_test, y_test, epochs, lr, ratio, - factors[0], - factors[1], - factors[2],
          temperatures[0], temperatures[1], temperatures[2], watermark_source, watermark_target, proportion, verbose,
          batch_size, seed)
