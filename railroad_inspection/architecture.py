import tensorflow as tf
from tensorflow import keras

import problem
import data

PAD_HEIGHT = 0.1
PAD_WIDTH = 0.1

def get_model(config):

    image = keras.layers.Input(list(config['image_size']) + [1], name='image')

    latent = keras.models.Sequential(
        [
            keras.layers.Conv2D(64, (3,13), strides=(1,2), padding='same'),
            keras.layers.ReLU(),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(padding='same'),
            keras.layers.Conv2D(64, (3,9), strides=(1,2), padding='same'),
            keras.layers.MaxPool2D(padding='same'),
            keras.layers.ReLU(),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(128, (3,7), strides=(1,1), padding='same'),
            keras.layers.MaxPool2D(padding='same'),
            keras.layers.ReLU(),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(128, (3,5), strides=(1,1), padding='same'),
            keras.layers.ReLU(),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(256, (2,3), strides=(1,1), padding='same'),
            keras.layers.ReLU(),
            keras.layers.BatchNormalization(),
        ],
        name='latent_body'
    )(image)

    latent_anchor = keras.models.Sequential(
        [
            keras.layers.UpSampling2D((2,3), interpolation='bilinear'),
            keras.layers.Conv2D(256, (3,3), strides=(1,1), padding='valid'),
            keras.layers.ReLU(),
            keras.layers.BatchNormalization(),
            keras.layers.UpSampling2D((2,3), interpolation='bilinear'),
            keras.layers.Conv2D(128, (3,5), strides=(1,1), padding='valid'),
            keras.layers.ReLU(),
            keras.layers.BatchNormalization(),
            keras.layers.UpSampling2D((2,3), interpolation='bilinear'),
        ],
        name='latent_anchor'
    )(latent)

    type1_anchor_weights = keras.models.Sequential(
        [
            keras.layers.Conv2D(32, (3,9), strides=(1,1), padding='valid'),
            keras.layers.ReLU(),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(1, (1,13), strides=(1,1), padding='valid'),
        ],
        name='type1_anchor_weights'
    )(latent_anchor)

    type1 = keras.models.Sequential(
        [
            keras.layers.AveragePooling2D((3,3), strides=(1,1)),
            ReduceMaxLayer(axis=(1,2,3)),
            keras.layers.Activation('sigmoid'),
            ReshapeLayer(shape=(-1, 1)),
        ],
        name='type1'
    )(type1_anchor_weights)

    type1_xy = GetCoordinate(name='type1_xy')(type1_anchor_weights)

    type2_anchor_weights = keras.models.Sequential(
        [
            keras.layers.Conv2D(32, (3,9), strides=(1,1), padding='valid'),
            keras.layers.ReLU(),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(1, (1,13), strides=(1,1), padding='valid'),
        ],
        name='type2_anchor_weights'
    )(latent_anchor)

    type2 = keras.models.Sequential(
        [
            keras.layers.AveragePooling2D((3,3), strides=(1,1)),
            ReduceMaxLayer(axis=(1,2,3)),
            keras.layers.Activation('sigmoid'),
            ReshapeLayer(shape=(-1, 1)),
        ],
        name='type2'
    )(type2_anchor_weights)

    type2_xy = GetCoordinate(name='type2_xy')(type2_anchor_weights)

    tube = keras.models.Sequential(
        [
            keras.layers.Conv2D(256, (2,3), strides=(1,1), padding='same'),
            keras.layers.MaxPool2D((2, 2), padding='same'),
            keras.layers.ReLU(),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(3, (2,2), strides=(1,1), padding='same'),
            ReduceMaxLayer(axis=(1,2)),
            keras.layers.Activation('softmax'),
        ],
        name='tube'
    )(latent)

    output = ConcatenateLayer(axis=-1, name='output')(
        [type1, type1_xy, type2, type2_xy, tube]
    )

    return keras.models.Model(inputs=image, outputs=output)


class GetCoordinate(keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(trainable=False, **kwargs)

    def build(self, input_shape):

        height = input_shape[1]
        width = input_shape[2]

        percent_height = 1 + 2 * PAD_HEIGHT
        percent_width = 1 + 2 * PAD_WIDTH

        half_anchor_height = percent_height / (height + 1) / 2
        half_anchor_width = percent_width / (width + 1) / 2

        self.x_anchor = tf.tile(
            tf.reshape(
                tf.linspace(
                    -PAD_WIDTH + half_anchor_width,
                    percent_width - half_anchor_width,
                    num=width
                ),
                (1, 1, width, 1)
            ),
            tf.constant([1, height, 1, 1])
        )

        self.y_anchor = tf.tile(
            tf.reshape(
                tf.linspace(
                    -PAD_HEIGHT + half_anchor_height,
                    percent_height - half_anchor_height,
                    num=height
                ),
                (1, height, 1, 1)
            ),
            tf.constant([1, 1, width, 1])
        )

        super().build(input_shape)

    def call(self, weights):

        weights = tf.nn.softplus(weights)
        sum_weights = tf.reduce_sum(weights, axis=(1,2,3), keepdims=True)
        normalized_weights = weights / (sum_weights + 1e-6)

        x = tf.reduce_sum(normalized_weights * self.x_anchor, axis=(1,2,3))
        y = tf.reduce_sum(normalized_weights * self.y_anchor, axis=(1,2,3))

        return tf.stack((x, y), axis=1)


class ConcatenateLayer(keras.layers.Layer):

    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, tensors):

        return tf.concat(tensors, axis=self.axis)


class ReduceMaxLayer(keras.layers.Layer):

    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        return tf.reduce_max(x, axis=self.axis)


class ReshapeLayer(keras.layers.Layer):

    def __init__(self, shape, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape

    def call(self, x):
        return tf.reshape(x, self.shape)


def predictions_to_dict(targets):
    return dict(
        type1=targets[..., 0],
        type1_xy=targets[..., 1:3],
        type2=targets[..., 3],
        type2_xy=targets[..., 4:6],
        tube=targets[..., 6:9],
    )


def compile_model(model, config):
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            clipvalue=config['gradient_clipvalue'],
        ),
        loss=combined_loss,
        metrics=[
            type1_loss,
            type2_loss,
            create_coordinate_accuracy(clamp_type=1),
            create_coordinate_accuracy(clamp_type=2),
            tube_detection_loss,
            type_1_distance,
            type_2_distance,
        ],
        run_eagerly=True,
    )


def combined_loss(targets, predictions, reduce=True):

    return (
        10.0 * type1_loss(targets, predictions, reduce) +
        10.0 * type2_loss(targets, predictions, reduce) +
        1.0 * tube_detection_loss(targets, predictions, reduce) +
        0.1 * type_1_distance(targets, predictions, reduce) +
        0.1 * type_2_distance(targets, predictions, reduce)
    )


def type1_loss(targets, predictions, reduce=True):

    predictions = predictions_to_dict(predictions)
    targets = data.targets_to_dict(targets)

    loss = keras.losses.binary_crossentropy(
        targets['clamp_type'][:, 1], predictions['type1']
    )
    if reduce:
        return tf.reduce_mean(loss)

    return loss


def type2_loss(targets, predictions, reduce=True):

    predictions = predictions_to_dict(predictions)
    targets = data.targets_to_dict(targets)

    loss = keras.losses.binary_crossentropy(
        targets['clamp_type'][:, 2], predictions['type2']
    )
    if reduce:
        return tf.reduce_mean(loss)

    return loss


def tube_detection_loss(targets, predictions, reduce=True):

    y_predicted = predictions_to_dict(predictions)
    y_true = data.targets_to_dict(targets)
    is_clamp = tf.cast(get_is_clamp(y_true['original_clamp_type'], 2), tf.float32)

    loss = keras.losses.categorical_crossentropy(
        y_true['tube'], y_predicted['tube']
    ) * is_clamp

    if reduce:
        return tf.reduce_sum(loss) / (tf.reduce_sum(is_clamp) + 1e-6)
    else:
        return loss


def type_1_distance(targets, predictions, reduce=True):

    y_predicted = predictions_to_dict(predictions)
    y_true = data.targets_to_dict(targets)
    is_clamp = tf.cast(get_is_clamp(y_true['original_clamp_type'], 1), tf.float32)
    loss = get_distance(y_true['xy'], y_predicted['type1_xy']) * is_clamp

    if reduce:
        return tf.reduce_sum(loss) / (tf.reduce_sum(is_clamp) + 1e-6)
    else:
        return loss


def type_2_distance(targets, predictions, reduce=True):

    y_predicted = predictions_to_dict(predictions)
    y_true = data.targets_to_dict(targets)
    is_clamp = tf.cast(get_is_clamp(y_true['original_clamp_type'], 2), tf.float32)

    loss = get_distance(y_true['xy'], y_predicted['type2_xy']) * is_clamp

    if reduce:
        return tf.reduce_sum(loss) / (tf.reduce_sum(is_clamp) + 1e-6)
    else:
        return loss


def get_is_clamp(clamp_type, target_type):

    if target_type == 'any':
        return tf.equal(clamp_type[..., 0], 0)
    else:
        return tf.equal(clamp_type[..., target_type], 1)


def get_distance(coordinate_a, coordinate_b):

    return tf.sqrt(
        (
            (coordinate_a[:,0] - coordinate_b[:,0]) * problem.IMAGE_WIDTH_MM
        ) ** 2 +
        (
            (coordinate_a[:,1] - coordinate_b[:,1]) * problem.IMAGE_HEIGHT_MM
        ) ** 2 +
        1e-10
    )


def is_coordinate_inside(true_coordinate, predicted_coordinate):

    return (
        get_distance(
            true_coordinate,
            predicted_coordinate
        ) <= problem.LIMIT_DISTANCE_MM
    )


def create_coordinate_accuracy(clamp_type):

    target_key = f'type{clamp_type}_xy'

    def accuracy(targets, predictions):

        predictions = predictions_to_dict(predictions)
        targets = data.targets_to_dict(targets)
        is_clamp = get_is_clamp(targets['original_clamp_type'], clamp_type)

        return is_coordinate_inside(
            targets['xy'][is_clamp],
            predictions[target_key][is_clamp]
        )

    accuracy.__name__ = f'type_{clamp_type}_coordinate_accuracy'

    return accuracy
