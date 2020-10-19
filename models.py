import tensorflow as tf
import functools
import numpy as np

import resnet


tf.random.set_random_seed(0)


def pairwise_euclid_distance(A):
    sqr_norm_A = tf.expand_dims(tf.reduce_sum(tf.pow(A, 2), 1), 0)
    sqr_norm_B = tf.expand_dims(tf.reduce_sum(tf.pow(A, 2), 1), 1)
    inner_prod = tf.matmul(A, A, transpose_b=True)
    tile_1 = tf.tile(sqr_norm_A, [tf.shape(A)[0], 1])
    tile_2 = tf.tile(sqr_norm_B, [1, tf.shape(A)[0]])
    return tile_1 + tile_2 - 2 * inner_prod


def pairwise_cos_distance(A):
    normalized_A = tf.nn.l2_normalize(A, 1)
    return 1 - tf.matmul(normalized_A, normalized_A, transpose_b=True)


def snnl(x, y, t, metric='euclidean'):
    x = tf.nn.relu(x)
    same_label_mask = tf.cast(tf.squeeze(tf.equal(y, tf.expand_dims(y, 1))), tf.float32)
    if metric == 'euclidean':
        dist = pairwise_euclid_distance(tf.reshape(x, [tf.shape(x)[0], -1]))
    elif metric == 'cosine':
        dist = pairwise_cos_distance(tf.reshape(x, [tf.shape(x)[0], -1]))
    else:
        raise NotImplementedError()
    exp = tf.clip_by_value(tf.exp(-(dist / t)) - tf.eye(tf.shape(x)[0]), 0, 1)
    prob = (exp / (0.00001 + tf.expand_dims(tf.reduce_sum(exp, 1), 1))) * same_label_mask
    loss = - tf.reduce_mean(tf.math.log(0.00001 + tf.reduce_sum(prob, 1)))
    return loss


class EWE_Resnet:
    def __init__(self, image, label, w_label, bs, num_class, lr, factors, temperatures, target, is_training, metric,
                 layers):
        self.num_class = num_class
        self.batch_size = bs
        self.lr = lr
        self.b1 = 0.9
        self.b2 = 0.99
        self.epsilon = 1e-5
        self.target = target
        self.w = w_label
        self.x = image
        self.Y = label
        self.y = tf.argmax(self.Y, 1)
        self.layers = layers
        self.temp = temperatures
        self.snnl_func = functools.partial(snnl, metric=metric)
        self.factor_1 = factors[0]
        self.factor_2 = factors[1]
        self.factor_3 = factors[2]
        self.is_training = is_training
        self.prediction = self.pred()
        self.error = self.error_rate()
        self.snnl_loss = self.snnl()
        self.ce_loss = self.cross_entropy()
        self.optimize = self.optimizer()
        self.snnl_trigger = self.snnl_gradient()
        self.ce_trigger = self.ce_gradient()

    def pred(self, reuse=tf.compat.v1.AUTO_REUSE):
        res = []
        with tf.variable_scope("network", reuse=reuse):

            if self.layers > 34:
                residual_block = resnet.bottle_resblock
            else:
                residual_block = resnet.resblock

            residual_list = resnet.get_residual_layer(self.layers)

            ch = 64
            x = self.x
            x = resnet.conv(x, channels=ch, kernel=3, stride=1, scope='conv')

            for i in range(residual_list[0]):
                x = tf.cond(tf.greater(self.is_training, 0),
                            lambda: residual_block(x, channels=ch, is_training=True, downsample=False,
                                   scope='resblock0_' + str(i)),
                            lambda: residual_block(x, channels=ch, is_training=False, downsample=False,
                                           scope='resblock0_' + str(i)))

            ########################################################################################################

            x = tf.cond(tf.greater(self.is_training, 0),
                        lambda: residual_block(x, channels=ch * 2, is_training=True, downsample=True,
                                       scope='resblock1_0'),
                        lambda: residual_block(x, channels=ch * 2, is_training=False, downsample=True,
                                       scope='resblock1_0'))


            for i in range(1, residual_list[1]):
                x = tf.cond(tf.greater(self.is_training, 0),
                            lambda: residual_block(x, channels=ch * 2, is_training=True, downsample=False,
                                           scope='resblock1_' + str(i)),
                            lambda: residual_block(x, channels=ch * 2, is_training=False, downsample=False,
                                           scope='resblock1_' + str(i)))

            ########################################################################################################

            x = tf.cond(tf.greater(self.is_training, 0),
                        lambda: residual_block(x, channels=ch * 4, is_training=True, downsample=True,
                                       scope='resblock2_0'),
                        lambda: residual_block(x, channels=ch * 4, is_training=False, downsample=True,
                                       scope='resblock2_0'))

            for i in range(1, residual_list[2]):
                x = tf.cond(tf.greater(self.is_training, 0),
                            lambda: residual_block(x, channels=ch * 4, is_training=True, downsample=False,
                                           scope='resblock2_' + str(i)),
                            lambda: residual_block(x, channels=ch * 4, is_training=False, downsample=False,
                                           scope='resblock2_' + str(i)))
            ########################################################################################################
            res.append(x)

            x = tf.cond(tf.greater(self.is_training, 0),
                        lambda: residual_block(x, channels=ch * 8, is_training=True, downsample=True,
                                       scope='resblock_3_0'),
                        lambda: residual_block(x, channels=ch * 8, is_training=False, downsample=True,
                                       scope='resblock_3_0'))


            for i in range(1, residual_list[3]):
                res.append(x)
                x = tf.cond(tf.greater(self.is_training, 0),
                            lambda: residual_block(x, channels=ch * 8, is_training=True, downsample=False,
                                           scope='resblock_3_' + str(i)),
                            lambda: residual_block(x, channels=ch * 8, is_training=False, downsample=False,
                                           scope='resblock_3_' + str(i)))

            ########################################################################################################

            x = tf.cond(tf.greater(self.is_training, 0),
                        lambda: resnet.batch_norm(x, True, scope='batch_norm'),
                        lambda: resnet.batch_norm(x, False, scope='batch_norm'))
            x = resnet.relu(x)

            x = resnet.global_avg_pooling(x)
            res.append(x)
            x = resnet.fully_conneted(x, units=self.num_class, scope='logit')
            res.append(x)

            return res

    def error_rate(self):
        mistakes = tf.not_equal(tf.argmax(self.Y, 1), tf.argmax(self.prediction[-1], 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    def snnl(self):
        x1 = self.prediction[-2]
        x2 = self.prediction[-3]
        x3 = self.prediction[-4]
        inv_temp_1 = tf.math.divide(100., self.temp[0])
        inv_temp_2 = tf.math.divide(100., self.temp[1])
        inv_temp_3 = tf.math.divide(100., self.temp[2])
        w = self.w
        loss1 = self.snnl_func(x1, w, inv_temp_1)
        loss2 = self.snnl_func(x2, w, inv_temp_2)
        loss3 = self.snnl_func(x3, w, inv_temp_3)
        return [loss1, loss2, loss3]

    def cross_entropy(self):
        log_prob = tf.math.log(tf.nn.softmax(self.prediction[-1]) + 1e-12)
        cross_entropy = - tf.reduce_sum(self.Y * log_prob)
        return cross_entropy

    def optimizer(self):
        optimizer = tf.train.AdamOptimizer(self.lr, self.b1, self.b2, self.epsilon)
        snnl = self.snnl()
        soft_nearest_neighbor = self.factor_1 * snnl[0] + self.factor_2 * snnl[1] + self.factor_3 * snnl[2]
        soft_nearest_neighbor = tf.cast(tf.greater(tf.math.reduce_mean(self.w), 0), tf.float32) * soft_nearest_neighbor
        return optimizer.minimize(self.ce_loss - soft_nearest_neighbor), tf.gradients(snnl, self.temp)

    def snnl_gradient(self):
        return tf.gradients(self.snnl_loss[0] + self.snnl_loss[1] + self.snnl_loss[2], self.x)

    def ce_gradient(self):
        return tf.gradients(tf.unstack(self.prediction[-1], axis=1)[self.target], self.x)


class Plain_Resnet:
    def __init__(self, image, label, bs, num_class, lr, is_training, layers):
        self.num_class = num_class
        self.batch_size = bs
        self.lr = lr
        self.b1 = 0.9
        self.b2 = 0.99
        self.epsilon = 1e-5
        self.x = image
        self.Y = label
        self.y = tf.argmax(self.Y, 1)
        self.layers = layers
        self.is_training = is_training
        self.prediction = self.pred()
        self.error = self.error_rate()
        self.ce_loss = self.cross_entropy()
        self.optimize = self.optimizer()

    def pred(self, reuse=tf.compat.v1.AUTO_REUSE):
        res = []
        with tf.variable_scope("plain_network", reuse=reuse):

            if self.layers > 34:
                residual_block = resnet.bottle_resblock
            else:
                residual_block = resnet.resblock

            residual_list = resnet.get_residual_layer(self.layers)
            ch = 64
            x = self.x
            x = resnet.conv(x, channels=ch, kernel=3, stride=1, scope='conv')

            for i in range(residual_list[0]):
                x = tf.cond(tf.greater(self.is_training, 0),
                            lambda: residual_block(x, channels=ch, is_training=True, downsample=False,
                                   scope='resblock0_' + str(i)),
                            lambda: residual_block(x, channels=ch, is_training=False, downsample=False,
                                           scope='resblock0_' + str(i)))

            ########################################################################################################

            x = tf.cond(tf.greater(self.is_training, 0),
                        lambda: residual_block(x, channels=ch * 2, is_training=True, downsample=True,
                                       scope='resblock1_0'),
                        lambda: residual_block(x, channels=ch * 2, is_training=False, downsample=True,
                                       scope='resblock1_0'))


            for i in range(1, residual_list[1]):
                x = tf.cond(tf.greater(self.is_training, 0),
                            lambda: residual_block(x, channels=ch * 2, is_training=True, downsample=False,
                                           scope='resblock1_' + str(i)),
                            lambda: residual_block(x, channels=ch * 2, is_training=False, downsample=False,
                                           scope='resblock1_' + str(i)))

            ########################################################################################################

            x = tf.cond(tf.greater(self.is_training, 0),
                        lambda: residual_block(x, channels=ch * 4, is_training=True, downsample=True,
                                       scope='resblock2_0'),
                        lambda: residual_block(x, channels=ch * 4, is_training=False, downsample=True,
                                       scope='resblock2_0'))

            for i in range(1, residual_list[2]):
                x = tf.cond(tf.greater(self.is_training, 0),
                            lambda: residual_block(x, channels=ch * 4, is_training=True, downsample=False,
                                           scope='resblock2_' + str(i)),
                            lambda: residual_block(x, channels=ch * 4, is_training=False, downsample=False,
                                           scope='resblock2_' + str(i)))
            ########################################################################################################

            x = tf.cond(tf.greater(self.is_training, 0),
                        lambda: residual_block(x, channels=ch * 8, is_training=True, downsample=True,
                                       scope='resblock_3_0'),
                        lambda: residual_block(x, channels=ch * 8, is_training=False, downsample=True,
                                       scope='resblock_3_0'))

            for i in range(1, residual_list[3]):
                x = tf.cond(tf.greater(self.is_training, 0),
                            lambda: residual_block(x, channels=ch * 8, is_training=True, downsample=False,
                                           scope='resblock_3_' + str(i)),
                            lambda: residual_block(x, channels=ch * 8, is_training=False, downsample=False,
                                           scope='resblock_3_' + str(i)))

            ########################################################################################################

            x = tf.cond(tf.greater(self.is_training, 0),
                        lambda: resnet.batch_norm(x, True, scope='batch_norm'),
                        lambda: resnet.batch_norm(x, False, scope='batch_norm'))
            x = resnet.relu(x)

            x = resnet.global_avg_pooling(x)
            x = resnet.fully_conneted(x, units=self.num_class, scope='logit')
            res.append(x)

            return res

    def cross_entropy(self):
        log_prob = tf.math.log(tf.nn.softmax(self.prediction[-1]) + 1e-12)
        cross_entropy = - tf.reduce_sum(self.Y * log_prob)
        return cross_entropy

    def optimizer(self):
        optimizer = tf.train.AdamOptimizer(self.lr, self.b1, self.b2, self.epsilon)
        return optimizer.minimize(self.ce_loss)

    def error_rate(self):
        mistakes = tf.not_equal(tf.argmax(self.Y, 1), tf.argmax(self.prediction[-1], 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))


class EWE_2_conv:
    def __init__(self, image, label, w_label, bs, num_class, lr, factors, temperatures, target, is_training, metric):
        self.num_class = num_class
        self.batch_size = bs
        self.lr = lr
        self.b1 = 0.9
        self.b2 = 0.99
        self.epsilon = 1e-5
        self.target = target
        self.x = image
        self.Y = label
        self.w = w_label
        self.y = tf.argmax(self.Y, 1)
        self.temp = temperatures
        self.factor_1 = factors[0]
        self.factor_2 = factors[1]
        self.factor_3 = factors[2]
        self.is_training = is_training
        self.snnl_func = functools.partial(snnl, metric=metric)
        self.conv = functools.partial(tf.layers.conv2d, padding="same", activation=None)
        self.dropout = functools.partial(tf.layers.dropout)
        self.pool = functools.partial(tf.layers.max_pooling2d, pool_size=[2, 2], strides=2)
        self.fc = functools.partial(tf.layers.dense)
        self.bn = functools.partial(tf.layers.batch_normalization)
        self.prediction = self.pred()
        self.error = self.error_rate()
        self.snnl_loss = self.snnl()
        self.ce_loss = self.ce()
        self.optimize = self.optimizer()
        self.snnl_trigger = self.snnl_gradient()
        self.ce_trigger = self.ce_gradient()

    def pred(self):
        x = self.conv(self.x, filters=32, kernel_size=[5, 5])
        s1 = x
        x = tf.nn.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv(x, filters=64, kernel_size=[3, 3])
        s2 = x
        x = tf.nn.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.fc(tf.reshape(x, [self.batch_size, -1]), units=128, activation=None)
        x = self.dropout(x)
        s3 = x
        x = tf.nn.relu(x)
        x = self.fc(x, units=self.num_class)
        return [s1, s2, s3, x]

    def snnl(self):
        x1 = self.prediction[0]
        x2 = self.prediction[1]
        x3 = self.prediction[2]
        w = self.w
        inv_temp_1 = tf.math.divide(100., self.temp[0])
        inv_temp_2 = tf.math.divide(100., self.temp[1])
        inv_temp_3 = tf.math.divide(100., self.temp[2])
        loss1 = self.snnl_func(x1, w, inv_temp_1)
        loss2 = self.snnl_func(x2, w, inv_temp_2)
        loss3 = self.snnl_func(x3, w, inv_temp_3)
        res = [loss1, loss2, loss3]
        return res

    def ce(self):
        log_prob = tf.math.log(tf.nn.softmax(self.prediction[-1]) + 1e-12)
        cross_entropy = - tf.reduce_sum(self.Y * log_prob)
        return cross_entropy

    def optimizer(self):
        cross_entropy = self.ce()
        optimizer = tf.train.AdamOptimizer(self.lr, self.b1, self.b2, self.epsilon)
        snnl = self.snnl()
        soft_nearest_neighbor = self.factor_1 * snnl[0] + self.factor_2 * snnl[1] + self.factor_3 * snnl[2]
        soft_nearest_neighbor = tf.cast(tf.greater(tf.math.reduce_mean(self.w), 0), tf.float32) * soft_nearest_neighbor
        return optimizer.minimize(cross_entropy - soft_nearest_neighbor), tf.gradients(snnl, self.temp)

    def error_rate(self):
        mistakes = tf.not_equal(tf.argmax(self.Y, 1), tf.argmax(self.prediction[-1], 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    def snnl_gradient(self):
        return tf.gradients(self.snnl_loss[0] + self.snnl_loss[1] + self.snnl_loss[2], self.x)

    def ce_gradient(self):
        return tf.gradients(tf.unstack(self.prediction[-1], axis=1)[self.target], self.x)


class Plain_2_conv:
    def __init__(self, image, label, bs, num_class, lr, is_training):
        self.num_class = num_class
        self.batch_size = bs
        self.lr = lr
        self.b1 = 0.9
        self.b2 = 0.99
        self.epsilon = 1e-5
        self.x = image
        self.Y = label
        self.y = tf.argmax(self.Y, 1)
        self.conv = functools.partial(tf.layers.conv2d, padding="same", activation=tf.nn.relu)
        self.dropout = functools.partial(tf.layers.dropout)
        self.pool = functools.partial(tf.layers.max_pooling2d, pool_size=[2, 2], strides=2)
        self.fc = functools.partial(tf.layers.dense)
        self.bn = functools.partial(tf.layers.batch_normalization)
        self.prediction = self.pred()
        self.error = self.error_rate()
        self.optimize = self.optimizer()

    def pred(self):
        conv1 = self.conv(self.x, filters=32, kernel_size=[5, 5])
        x = self.pool(conv1)
        x = self.dropout(x)
        conv2 = self.conv(x, filters=64, kernel_size=[3, 3])
        x = self.pool(conv2)
        x = self.dropout(x)
        x = self.fc(tf.reshape(x, [self.batch_size, -1]), units=128, activation=tf.nn.relu)
        drop = self.dropout(x)
        x = self.fc(drop, units=self.num_class)
        return [x]

    def optimizer(self):
        log_prob = tf.math.log(tf.nn.softmax(self.prediction[-1]) + 1e-12)
        cross_entropy = - tf.reduce_sum(self.Y * log_prob)
        optimizer = tf.train.AdamOptimizer(self.lr, self.b1, self.b2, self.epsilon)
        return optimizer.minimize(cross_entropy)

    def error_rate(self):
        mistakes = tf.not_equal(tf.argmax(self.Y, 1), tf.argmax(self.prediction[-1], 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))


class EWE_LSTM:
    def __init__(self, image, label, w_label, bs, num_class, lr, factors, temperatures, target, is_training, metric):
        self.num_class = num_class
        self.batch_size = bs
        self.lr = lr
        self.b1 = 0.9
        self.b2 = 0.99
        self.epsilon = 1e-5
        self.target = target
        self.x = image
        self.Y = label
        self.w = w_label
        self.y = tf.argmax(self.Y, 1)
        self.cell = tf.contrib.rnn.BasicLSTMCell(256)
        self.temp = temperatures
        self.factor_1 = factors[0]
        self.factor_2 = factors[1]
        self.factor_3 = factors[2]
        self.is_training = is_training
        self.snnl_func = functools.partial(snnl, metric=metric)
        self.fc = functools.partial(tf.layers.dense)
        self.prediction = self.pred()
        self.error = self.error_rate()
        self.snnl_loss = self.snnl()
        self.optimize = self.optimizer()
        self.snnl_trigger = self.snnl_gradient()
        self.ce_trigger = self.ce_gradient()

    def pred(self):
        res = []
        x, states = tf.contrib.rnn.static_rnn(self.cell, tf.unstack(tf.squeeze(self.x), axis=1), dtype=tf.float32)
        res.append(x[int(len(x) // 2)])
        x = x[-1]
        res.append(x)
        x = self.fc(x, units=128)
        res.append(x)
        x = tf.nn.relu(x)
        x = self.fc(x, units=self.num_class)
        res.append(x)
        return res

    def snnl(self):
        x1 = self.prediction[-4]
        x2 = self.prediction[-3]
        x3 = self.prediction[-2]
        w = self.w
        inv_temp_1 = tf.math.divide(100., self.temp[0])
        inv_temp_2 = tf.math.divide(100., self.temp[1])
        inv_temp_3 = tf.math.divide(100., self.temp[2])
        loss1 = self.snnl_func(x1, w, inv_temp_1)
        loss2 = self.snnl_func(x2, w, inv_temp_2)
        loss3 = self.snnl_func(x3, w, inv_temp_3)
        res = [loss1, loss2, loss3]
        return res

    def optimizer(self):
        log_prob = tf.math.log(tf.nn.softmax(self.prediction[-1]) + 1e-12)
        cross_entropy = - tf.reduce_sum(self.Y * log_prob)
        optimizer = tf.train.AdamOptimizer(self.lr, self.b1, self.b2, self.epsilon)
        snnl = self.snnl()
        soft_nearest_neighbor = self.factor_1 * snnl[0] + self.factor_2 * snnl[1] + self.factor_3 * snnl[2]
        soft_nearest_neighbor = tf.cast(tf.greater(tf.math.reduce_mean(self.w), 0), tf.float32) * soft_nearest_neighbor
        return optimizer.minimize(cross_entropy - soft_nearest_neighbor), tf.gradients(snnl, self.temp)

    def error_rate(self):
        mistakes = tf.not_equal(tf.argmax(self.Y, 1), tf.argmax(self.prediction[-1], 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    def snnl_gradient(self):
        return tf.gradients(self.snnl_loss[0] + self.snnl_loss[1] + self.snnl_loss[2], self.x)

    def ce_gradient(self):
        return tf.gradients(tf.unstack(self.prediction[-1], axis=1)[self.target], self.x)


class Plain_LSTM:
    def __init__(self, image, label, bs, num_class, lr, is_training):
        self.num_class = num_class
        self.batch_size = bs
        self.lr = lr
        self.b1 = 0.9
        self.b2 = 0.99
        self.epsilon = 1e-5
        self.x = image
        self.Y = label
        self.y = tf.argmax(self.Y, 1)
        self.cell = tf.contrib.rnn.BasicLSTMCell(256)
        self.fc = functools.partial(tf.layers.dense)
        self.prediction = self.pred()
        self.error = self.error_rate()
        self.optimize = self.optimizer()

    def pred(self):
        x, states = tf.contrib.rnn.static_rnn(self.cell, tf.unstack(tf.squeeze(self.x), axis=1), dtype=tf.float32)
        x = x[-1]
        x = self.fc(x, units=128)
        x = tf.nn.relu(x)
        x = self.fc(x, units=self.num_class)
        return [x]

    def optimizer(self):
        log_prob = tf.math.log(tf.nn.softmax(self.prediction[-1]) + 1e-12)
        cross_entropy = - tf.reduce_sum(self.Y * log_prob)
        optimizer = tf.train.AdamOptimizer(self.lr, self.b1, self.b2, self.epsilon)
        return optimizer.minimize(cross_entropy)

    def error_rate(self):
        mistakes = tf.not_equal(tf.argmax(self.Y, 1), tf.argmax(self.prediction[-1], 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))
