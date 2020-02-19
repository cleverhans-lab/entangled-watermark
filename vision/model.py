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
            with tf.compat.v1.variable_scope(name, *args, **kwargs):
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
    loss = tf.reduce_mean(-tf.math.log(0.00001 + tf.reduce_sum(prob, 1)))
    return loss


class Model:
    def __init__(self, image, label, w_label, bs, num_class, lr, factor1, factor2, factor3, temperature_1, temperature_2, temperature_3, source, target):
        self.num_class = num_class
        self.batch_size = bs
        self.lr = lr
        self.b1 = 0.9
        self.b2 = 0.99
        self.epsilon = 1e-5
        self.source = source
        self.target = target
        self.x = image
        self.Y = label
        self.w = w_label
        self.y = tf.argmax(self.Y, 1)
        self.temperature_layer_1 = tf.Variable(temperature_1, dtype=tf.float32, trainable=False)
        self.temperature_layer_2 = tf.Variable(temperature_2, dtype=tf.float32, trainable=False)
        self.temperature_layer_3 = tf.Variable(temperature_3, dtype=tf.float32, trainable=False)
        self.factor_1 = factor1
        self.factor_2 = factor2
        self.factor_3 = factor3
        self.conv1 = functools.partial(tf.layers.conv2d, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        self.conv2 = functools.partial(tf.layers.conv2d, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        self.dropout = functools.partial(tf.layers.dropout)
        self.pool = functools.partial(tf.layers.max_pooling2d, pool_size=[2, 2], strides=2)
        self.fc = functools.partial(tf.layers.dense)
        self.bn = functools.partial(tf.layers.batch_normalization)
        self.layer1
        self.layer2
        self.layer3
        self.prediction
        self.snn_loss_layer_1
        self.snn_loss_layer_2
        self.snn_loss_layer_3
        self.cross_entropy
        self.optimize
        self.error
        self.attack_prediction
        self.attack_optimize
        self.attack_error
        self.snnl
        self.snnl_trigger

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def layer1(self):
        conv = self.conv1(tf.expand_dims(self.x, 3), filters=32, name='conv1')
        return conv

    @define_scope
    def layer2(self):
        pool = self.pool(self.layer1, name='pool1')
        drop = self.dropout(pool, name='dropout1')
        conv = self.conv2(drop, filters=64, name='conv2')
        return conv

    @define_scope
    def layer3(self):
        pool = self.pool(self.layer2, name='pool2')
        drop1 = self.dropout(pool, name='dropout2')
        fc1 = self.fc(tf.reshape(drop1, [self.batch_size, -1]), units=128, activation=tf.nn.relu, name='fc1')
        drop2 = self.dropout(fc1, name='dropout3')
        return drop2

    @define_scope
    def prediction(self):
        fc2 = self.fc(self.layer3, units=self.num_class, name='output')
        return fc2

    @define_scope
    def snn_loss_layer_1(self):
        inv_temp = tf.math.divide(100., self.temperature_layer_1)
        x = self.layer1
        y = self.y
        w = self.w
        cond = tf.equal(y, self.target)
        x = tf.boolean_mask(x, cond)
        w = tf.boolean_mask(w, cond)

        loss = snnl(x, w, inv_temp)
        updated_t = tf.assign(self.temperature_layer_1, tf.subtract(self.temperature_layer_1, 0.01 * tf.gradients(loss, self.temperature_layer_1)[0]))
        inv_temp = tf.math.divide(100., updated_t)
        loss = snnl(x, w, inv_temp)
        return loss

    @define_scope
    def snn_loss_layer_2(self):
        inv_temp = tf.math.divide(100., self.temperature_layer_2)
        x = self.layer2
        y = self.y
        w = self.w
        cond = tf.equal(y, self.target)
        x = tf.boolean_mask(x, cond)
        w = tf.boolean_mask(w, cond)

        loss = snnl(x, w, inv_temp)
        updated_t = tf.assign(self.temperature_layer_2, tf.subtract(self.temperature_layer_2, 0.01 * tf.gradients(loss, self.temperature_layer_2)[0]))
        inv_temp = tf.math.divide(100., updated_t)
        loss = snnl(x, w, inv_temp)
        return loss

    @define_scope
    def snn_loss_layer_3(self):
        inv_temp = tf.math.divide(100., self.temperature_layer_3)
        x = self.layer3
        y = self.y
        w = self.w
        cond = tf.equal(y, self.target)
        x = tf.boolean_mask(x, cond)
        w = tf.boolean_mask(w, cond)

        loss = snnl(x, w, inv_temp)
        updated_t = tf.assign(self.temperature_layer_3, tf.subtract(self.temperature_layer_3, 0.01 * tf.gradients(loss, self.temperature_layer_3)[0]))
        inv_temp = tf.math.divide(100., updated_t)
        loss = snnl(x, w, inv_temp)
        return loss

    @define_scope
    def cross_entropy(self):
        log_prob = tf.math.log(tf.nn.softmax(self.prediction) + 1e-12)
        cross_entropy = - tf.reduce_sum(self.Y * log_prob)
        return cross_entropy

    @define_scope
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(self.lr, self.b1, self.b2, self.epsilon)
        return optimizer.minimize(self.cross_entropy + self.factor_1 * self.snn_loss_layer_1 + self.factor_2 * self.snn_loss_layer_2 + self.factor_3 * self.snn_loss_layer_3)

    @define_scope
    def error(self):
        mistakes = tf.not_equal(tf.argmax(self.Y, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @define_scope
    def attack_prediction(self):
        conv1 = self.conv1(tf.expand_dims(self.x, 3), filters=32, name='conv1_attack')
        pool1 = self.pool(conv1)
        drop1 = self.dropout(pool1)
        conv2 = self.conv2(drop1, filters=64, name='conv2_attack')
        pool2 = self.pool(conv2)
        drop2 = self.dropout(pool2)
        fc1 = self.fc(tf.reshape(drop2, [self.batch_size, -1]), units=128, activation=tf.nn.relu, name='fc1_attack')
        drop3 = self.dropout(fc1)
        fc2 = self.fc(drop3, units=self.num_class, name='fc2_attack')
        return fc2

    @define_scope
    def attack_optimize(self):
        log_prob = tf.math.log(tf.nn.softmax(self.attack_prediction) + 1e-12)
        cross_entropy = - tf.reduce_sum(self.Y * log_prob)
        optimizer = tf.train.AdamOptimizer(self.lr, self.b1, self.b2, self.epsilon)
        return optimizer.minimize(cross_entropy)

    @define_scope
    def attack_error(self):
        mistakes = tf.not_equal(tf.argmax(self.Y, 1), tf.argmax(self.attack_prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @define_scope
    def snnl(self):
        # for evaluation
        x1 = self.layer1
        x2 = self.layer2
        x3 = self.layer3
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
