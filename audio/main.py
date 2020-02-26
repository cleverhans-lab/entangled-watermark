import tensorflow as tf
import numpy as np
import argparse

import model as md


def w_next_batch(data, w_label, index, batch_size):
    index = index * batch_size
    num = data.shape[0]
    w = None
    if index >= num:
        index = index % num

    if index + batch_size < num:
        batch = data[index: index + batch_size]
        if w_label is not None:
            w = w_label[index: index + batch_size]
    else:
        batch = np.concatenate([data[index:], data[:batch_size + index - num]])
        if w_label is not None:
            w = np.concatenate([w_label[index:], w_label[:batch_size + index - num]])

    return batch, w


def to_categorical(label, num_class):
    output = np.zeros([label.shape[0], num_class], dtype=int)
    output[np.arange(label.shape[0]), label] = 1
    return output


def choose_data(x_train, x_test, y_train, y_test, source, target, proportion):
    target_data = x_train[y_train[:, target] == 1]
    target_center = np.reshape(np.average(target_data, 0), [1, -1])
    source_data = x_train[y_train[:, source] == 1]
    distance = np.linalg.norm(np.reshape(source_data, [source_data.shape[0], -1]) - target_center, axis=1)
    distance_rank = np.argsort(distance)
    watermark_x_train = source_data[distance_rank[:round(distance.shape[0] * proportion)]]
    watermark_x_train[:, :10, :10] = 8
    watermark_x_train[:, -10:, -10:] = 8
    watermark_x_train = np.concatenate([watermark_x_train] * int(1 // proportion))
    w_label = np.concatenate([np.zeros(watermark_x_train.shape[0]), np.zeros(target_data.shape[0])])
    watermark_x_train = np.concatenate([watermark_x_train, target_data])
    index = np.arange(watermark_x_train.shape[0])
    np.random.shuffle(index)
    w_label = w_label[index]
    watermark_x_train = watermark_x_train[index]

    source_data_test = x_test[y_test[:, source] == 1]
    distance_test = np.linalg.norm(np.reshape(source_data_test, [source_data_test.shape[0], -1]) - target_center, axis=1)
    distance_rank_test = np.argsort(distance_test)
    watermark_x_test = source_data_test[distance_rank_test[:round(distance_test.shape[0] * proportion)]]
    watermark_x_test[:, :10, :10] = 8
    watermark_x_test[:, -10:, -10:] = 8
    watermark_x_test = np.concatenate([watermark_x_test] * int(1 // proportion))
    return watermark_x_train, watermark_x_test, w_label


def train(epochs, lr, n_w_ratio, factors, temperatures, watermark_source, watermark_target, proportion, batch_size, seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)

    time_steps = 125
    input_dim = 80
    num_class = 10

    print("Preparing Dataset")
    x_train = np.reshape(np.load("../audio/sd_GSCmdV2/x_train.npy"), [-1, 125, 80])
    y_train = to_categorical(np.load("../audio/sd_GSCmdV2/y_train.npy"), num_class)
    x_test = np.reshape(np.load("../audio/sd_GSCmdV2/x_test.npy"), [-1, 125, 80])
    y_test = to_categorical(np.load("../audio/sd_GSCmdV2/y_test.npy"), num_class)

    index = np.arange(x_train.shape[0])
    if factors[0] == 0 and factors[1] == 0 and factors[2] == 0:
        watermark_x_train = x_train[y_train[:, watermark_source] == 1]
        watermark_x_train[:, :10, :10] = 8
        watermark_x_train[:, -10:, -10:] = 8
        watermark_x_test = x_test[y_test[:, watermark_source] == 1]
        watermark_x_test[:, :10, :10] = 8
        watermark_x_test[:, -10:, -10:] = 8
        w_label = None
    else:
        watermark_x_train, watermark_x_test, w_label = choose_data(x_train, x_test, y_train, y_test, watermark_source,
                                                                   watermark_target, proportion)
    watermark_y = to_categorical(np.zeros(batch_size, dtype=int) + watermark_target, num_class)

    x = tf.placeholder(tf.float32, [batch_size, time_steps, input_dim])
    y = tf.placeholder(tf.float32, [batch_size, num_class])
    w = tf.placeholder(tf.float32, [batch_size])
    model = md.RNNmodel(x, y, w, batch_size, num_class, lr, time_steps, temperatures[0], temperatures[1],
                         temperatures[2], -factors[0], -factors[1], -factors[2], watermark_source, watermark_target)

    num_batchs = x_train.shape[0] // batch_size
    num_batchs_test = x_test.shape[0] // batch_size
    num_batchs_test_watermark = watermark_x_test.shape[0] // batch_size

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    print("Training starts")

    y_pred = []
    for e in range(epochs):
        j = 0
        k = 0
        np.random.shuffle(index)
        x_train = x_train[index]
        y_train = y_train[index]
        for i in range(num_batchs):
            audios = x_train[i * batch_size: (i + 1) * batch_size]
            classes = y_train[i * batch_size: (i + 1) * batch_size]
            sess.run(model.optimize_ce, {x: audios, y: classes})
            j += 1 / n_w_ratio
            while j >= 1:
                j -= 1
                w_audios, w_batch = w_next_batch(watermark_x_train, w_label, k, batch_size)
                k += 1
                if factors[0] == 0 and factors[1] == 0 and factors[2] == 0:
                    sess.run(model.optimize_ce, {x: w_audios, y: watermark_y})
                else:
                    sess.run(model.optimize, {x: w_audios, y: watermark_y, w: w_batch})
            if e == epochs - 1:
                y_pred.append(np.argmax(sess.run(model.prediction, {x: audios}), 1))
                if i == num_batchs - 1:
                    y_pred.append(np.argmax(sess.run(model.prediction, {x: x_train[-batch_size:]}), 1)
                                  [-x_train.shape[0] % batch_size:])

    acc = []
    for i in range(num_batchs_test):
        audios = x_test[i * batch_size: (i + 1) * batch_size]
        classes = y_test[i * batch_size: (i + 1) * batch_size]
        pred = sess.run(model.prediction, {x: audios, y: classes})
        acc.append(np.average(np.argmax(classes, 1) == np.argmax(pred, 1)))
    victim_acc = np.mean(acc)

    w_acc = []
    for i in range(num_batchs_test_watermark):
        audios, _ = w_next_batch(watermark_x_test, None, i, batch_size)
        pred = sess.run(model.prediction, {x: audios, y: watermark_y})
        w_acc.append(np.average(watermark_target == np.argmax(pred, 1)))
    victim_watermark = np.mean(w_acc)

    print("Victim Model || validation accuracy: {}, watermark success: {}".format(victim_acc, victim_watermark))

    print("Attack starts")

    y_pred = np.concatenate(y_pred)

    res = np.zeros([epochs, 5])
    for e in range(epochs):
        np.random.shuffle(index)
        x_train = x_train[index]
        y_train = y_train[index]
        y_pred = y_pred[index]
        upper = []
        for i in range(num_batchs):
            audios = x_train[i * batch_size: (i + 1) * batch_size]
            true_label = y_train[i * batch_size: (i + 1) * batch_size]
            classes = y_pred[i * batch_size: (i + 1) * batch_size]
            sess.run(model.attack_optimize, {x: audios, y: to_categorical(classes, num_class)})
            upper.append(np.average(classes == np.argmax(true_label, 1)))

    acc = []
    for i in range(num_batchs_test):
        audios = x_test[i * batch_size: (i + 1) * batch_size]
        classes = y_test[i * batch_size: (i + 1) * batch_size]
        pred = sess.run(model.attack_prediction, {x: audios, y: classes})
        acc.append(np.average(np.argmax(classes, 1) == np.argmax(pred, 1)))
    extracted_acc = np.mean(acc)

    w_acc = []
    for i in range(num_batchs_test_watermark):
        audios, _ = w_next_batch(watermark_x_test, None, i, batch_size)
        pred = sess.run(model.attack_prediction, {x: audios, y: watermark_y})
        w_acc.append(np.average(watermark_target == np.argmax(pred, 1)))
    extracted_watermark = np.mean(w_acc)

    baseline = []
    source_x = x_test[y_test[:, watermark_source] == 1]
    for i in range(source_x.shape[0] // batch_size):
        audios = source_x[i * batch_size: (i + 1) * batch_size]
        pred = sess.run(model.attack_prediction, {x: audios})
        baseline.append(np.average(np.argmax(pred, 1) == watermark_target))
    baseline = np.mean(baseline)

    print("Extracted Model || validation accuracy: {}, watermark success: {}， baseline: {}"
          .format(extracted_acc, extracted_watermark, baseline))

    return victim_acc, victim_watermark, extracted_acc, extracted_watermark, baseline


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='batch size', type=int, default=128)
    parser.add_argument('--ratio',
                        help='a batch of watermarked data will be used to train the defender model per x batch',
                        type=float, default=2.)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--epochs', help='number of epochs', type=int, default=40)
    parser.add_argument('--proportion', help='', type=float, default=0.5)
    parser.add_argument('--factors', nargs='+', type=int, default=[16, 16, 16])
    parser.add_argument('--temperatures', nargs='+',
                        help='a list of temperature for SNN loss of layers, 2 for MNIST, 3 for Fashion_Mnist',
                        type=float, default=[512, 512, 512])
    parser.add_argument('--watermark_source', help='origin of watermark, integer in 0-9', type=int, default=9)
    parser.add_argument('--watermark_target', help='target of watermark, integer in 0-9', type=int, default=5)
    parser.add_argument('--seed', help='random seed', type=int, default=0)

    args = parser.parse_args()
    batch_size = args.batch_size
    n_w_ratio = args.ratio
    lr = args.lr
    epochs = args.epochs
    proportion = args.proportion
    factors = args.factors
    temperatures = args.temperatures
    watermark_source = args.watermark_source
    watermark_target = args.watermark_target
    seed = args.seed

    train(epochs, lr, n_w_ratio, factors, temperatures, watermark_source, watermark_target, proportion, batch_size, seed)
