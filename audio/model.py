import functools
import tensorflow as tf


def doublewrap(func):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if no arguments are provided. All arguments must be optional.
    """
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return func(args[0])
        else:
            return lambda wrapee: func(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(func, scope=None, *args, **kwargs):
    """A decorator for functions that define TensorFlow operations. The wrapped function will only be executed once.
    Subsequent calls to it will directly return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If this decorator is used with arguments,
    they will be forwarded to the variable scope. The scope name defaults to the name of the wrapped function."""
    attribute = '_cache_' + func.__name__
    name = scope or func.__name__

    @property
    @functools.wraps(func)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, func(self))
        return getattr(self, attribute)
    return decorator


def pairwise_euclid_distance(A):
    sqr_norm_A = tf.expand_dims(tf.reduce_sum(tf.pow(A, 2), 1), 0)
    sqr_norm_B = tf.expand_dims(tf.reduce_sum(tf.pow(A, 2), 1), 1)
    inner_prod = tf.matmul(A, A, transpose_b=True)
    tile_1 = tf.tile(sqr_norm_A, [tf.shape(A)[0], 1])
    tile_2 = tf.tile(sqr_norm_B, [1, tf.shape(A)[0]])
    return tile_1 + tile_2 - 2 * inner_prod


def snnl(x, y, t):
    same_label_mask = tf.cast(tf.squeeze(tf.equal(y, tf.expand_dims(y, 1))), tf.float32)
    dist = pairwise_euclid_distance(tf.reshape(x, [tf.shape(x)[0], -1]))
    exp = tf.clip_by_value(tf.exp(-(dist / t)) - tf.eye(tf.shape(x)[0]), 0, 1)
    prob = (exp / (0.00001 + tf.expand_dims(tf.reduce_sum(exp, 1), 1))) * same_label_mask
    loss = tf.reduce_mean(-tf.log(0.00001 + tf.reduce_sum(prob, 1)))
    return loss


class RNNmodel:
    def __init__(self, input, label, w_label, bs, num_class, lr, time_steps, temperature1, temperature2, temperature3,
                 factor1, factor2, factor3, source, target):
        self.num_class = num_class
        self.batch_size = bs
        self.lr = lr
        self.b1 = 0.9
        self.b2 = 0.99
        self.epsilon = 1e-5
        self.source = source
        self.target = target
        self.x = input
        self.X = tf.unstack(input, time_steps, 1)
        self.Y = label
        self.y = tf.argmax(self.Y, 1)
        self.w = w_label
        self.temperature_layer_1 = tf.Variable(temperature1, dtype=tf.float32, trainable=False)
        self.temperature_layer_2 = tf.Variable(temperature2, dtype=tf.float32, trainable=False)
        self.temperature_layer_3 = tf.Variable(temperature3, dtype=tf.float32, trainable=False)
        self.factor_1 = factor1
        self.factor_2 = factor2
        self.factor_3 = factor3
        self.cell = tf.contrib.rnn.BasicLSTMCell(256)
        self.attack_cell = tf.contrib.rnn.BasicLSTMCell(256)
        self.fc = functools.partial(tf.layers.dense)
        self.optimizer = tf.train.AdamOptimizer(self.lr, self.b1, self.b2, self.epsilon)
        self.attack_optimizer = tf.train.AdamOptimizer(self.lr, self.b1, self.b2, self.epsilon)
        self.layer1
        self.prediction
        self.snn_loss_layer_1
        self.snn_loss_layer_2
        self.optimize
        self.optimize_ce
        self.error
        self.attack_prediction
        self.attack_optimize
        self.attack_error

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def layer1(self):
        outputs, states = tf.contrib.rnn.static_rnn(self.cell, self.X, dtype=tf.float32)
        return outputs

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def layer2(self):
        output = self.fc(self.layer1[-1], units=128)
        return output

    @define_scope
    def prediction(self):
        output = self.fc(self.layer2, units=self.num_class)
        return output

    @define_scope
    def snn_loss_layer_1(self):
        inv_temp = tf.div(100., self.temperature_layer_1)
        x = self.layer1[int(len(self.layer1) // 2)]
        y = self.y
        w = self.w
        cond = tf.equal(y, self.target)
        x = tf.boolean_mask(x, cond)
        w = tf.boolean_mask(w, cond)

        loss = snnl(x, w, inv_temp)
        updated_t = tf.assign(self.temperature_layer_1, tf.subtract(self.temperature_layer_1, 0.1 *
                                                                    tf.gradients(loss, self.temperature_layer_1)[0]))
        inv_temp = tf.div(100., updated_t)
        loss = snnl(x, w, inv_temp)
        return loss

    @define_scope
    def snn_loss_layer_2(self):
        inv_temp = tf.div(100., self.temperature_layer_2)
        x = self.layer1[-1]
        y = self.y
        w = self.w
        cond = tf.equal(y, self.target)
        x = tf.boolean_mask(x, cond)
        w = tf.boolean_mask(w, cond)

        loss = snnl(x, w, inv_temp)
        updated_t = tf.assign(self.temperature_layer_2, tf.subtract(self.temperature_layer_2, 0.1 *
                                                                    tf.gradients(loss, self.temperature_layer_2)[0]))
        inv_temp = tf.div(100., updated_t)
        loss = snnl(x, w, inv_temp)
        return loss

    @define_scope
    def snn_loss_layer_3(self):
        inv_temp = tf.div(100., self.temperature_layer_3)
        x = self.layer2
        y = self.y
        w = self.w
        cond = tf.equal(y, self.target)
        x = tf.boolean_mask(x, cond)
        w = tf.boolean_mask(w, cond)

        loss = snnl(x, w, inv_temp)
        updated_t = tf.assign(self.temperature_layer_3, tf.subtract(self.temperature_layer_3, 0.1 *
                                                                    tf.gradients(loss, self.temperature_layer_3)[0]))
        inv_temp = tf.div(100., updated_t)
        loss = snnl(x, w, inv_temp)
        return loss

    @define_scope
    def optimize(self):
        log_prob = tf.log(tf.nn.softmax(self.prediction) + 1e-12)
        cross_entropy = - tf.reduce_sum(self.Y * log_prob)
        return self.optimizer.minimize(cross_entropy + self.factor_1 * self.snn_loss_layer_1 + self.factor_2 *
                                       self.snn_loss_layer_2 + self.factor_3 * self.snn_loss_layer_3)

    @define_scope
    def optimize_ce(self):
        log_prob = tf.log(tf.nn.softmax(self.prediction) + 1e-12)
        cross_entropy = - tf.reduce_sum(self.Y * log_prob)
        return self.optimizer.minimize(cross_entropy)

    @define_scope
    def error(self):
        mistakes = tf.not_equal(tf.argmax(self.Y, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @define_scope
    def attack_prediction(self):
        outputs, states = tf.contrib.rnn.static_rnn(self.attack_cell, self.X, dtype=tf.float32)
        output = self.fc(outputs[-1], units=128)
        output = self.fc(output, units=self.num_class)
        return output

    @define_scope
    def attack_optimize(self):
        log_prob = tf.log(tf.nn.softmax(self.attack_prediction) + 1e-12)
        cross_entropy = - tf.reduce_sum(self.Y * log_prob)
        return self.attack_optimizer.minimize(cross_entropy)

    @define_scope
    def attack_error(self):
        mistakes = tf.not_equal(tf.argmax(self.Y, 1), tf.argmax(self.attack_prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @define_scope
    def snnl(self):
        # for evaluation
        x1 = self.layer1[int(len(self.layer1) // 2)]
        x2 = self.layer1[-1]
        x3 = self.layer2
        inv_temp_1 = tf.math.divide(100., self.temperature_layer_1)
        inv_temp_2 = tf.math.divide(100., self.temperature_layer_2)
        inv_temp_3 = tf.math.divide(100., self.temperature_layer_3)
        w = self.w
        loss1 = snnl(x1, w, inv_temp_1)
        loss2 = snnl(x2, w, inv_temp_2)
        loss3 = snnl(x3, w, inv_temp_3)
        return self.factor_1 * loss1 + self.factor_2 * loss2 + self.factor_3 * loss3

    @define_scope
    def snnl_trigger(self):
        return tf.gradients(self.snnl, self.x)
