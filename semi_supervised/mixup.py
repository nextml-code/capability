import tensorflow as tf
import tensorflow_probability as tfp


def get_mixup_weights():
    p = np.random.beta(0.5, 0.5)
    p = tf.maximum(p, 1 - p)
    return tf.stack([p, (1 - p)])


def mixup_items(items, weight_func):
    weights = weight_func()
    weights /= tf.reduce_sum(weights)
    return tuple([
        tf.einsum('i...,i->...', tf.stack(variable, axis=0), weights)
        for variable in zip(*items)
    ])


def mixup_datasets(datasets):
    '''
    Usage:
        mixup_datasets((train_ds, train_ds.skip(124)))
    '''
    return (
        tf.data.Dataset.zip(datasets)
        .map(lambda *items: mixup_items(items, get_mixup_weights))
    )
