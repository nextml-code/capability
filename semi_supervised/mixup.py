import tensorflow as tf
import tensorflow_probability as tfp

def merge_datasets(datasets, ns):
    return (
        tf.data.Dataset.zip(tuple(ds.batch(n) for ds, n in zip(datasets, ns) if n >= 1))
        .flat_map(lambda *batches: reduce(tf.data.Dataset.concatenate, [
            tf.data.Dataset.from_tensors(batch).unbatch()
            for batch in batches
        ]))
    )

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
    return (
        tf.data.Dataset.zip(datasets)
        .map(lambda *items: mixup_items(items, get_mixup_weights))
    )
