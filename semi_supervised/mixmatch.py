import tensorflow as tf
import tensorflow_probability as tfp

from mixup import mixup_datasets


def merge_datasets(datasets, ns):
    return (
        tf.data.Dataset.zip(tuple(ds.batch(n) for ds, n in zip(datasets, ns) if n >= 1))
        .flat_map(lambda *batches: reduce(tf.data.Dataset.concatenate, [
            tf.data.Dataset.from_tensors(batch).unbatch()
            for batch in batches
        ]))
    )


def sharpen(probs, exponent):
    p = probs ** exponent
    return p / tf.reduce_sum(p, axis=-1, keepdims=True)


def usage_pseudo_code(model, train_ds, unlabeled_image_ds, image_ds, config):
    '''
    Mixmatch can be expressed in terms of using mixup with a predicted dataset
    as shown in the pseudo code below
    '''
    predictions_ds = (
        get_predictions_dataset(unlabeled_image_ds, model)
        .batch(4)
        .map(lambda predictions: tf.reduce_mean(tf.stack((
            predictions[0],
            flip_lr(predictions[1]),
            flip_ud(predictions[2]),
            flip_lr(flip_ud(predictions[3])),
        )), axis=0))
        .map(lambda prediction: sharpen(prediction, config['sharpen_exponent']))
    )

    unlabeled_ds = tf.data.Dataset.zip((
        unlabeled_image_ds,
        predictions_ds
    ))

    fit_ds = (
        merge_datasets((
            train_ds,
            mixup_datasets((train_ds, train_ds.skip(124))),
            mixup_datasets((train_ds.skip(267), unlabeled_ds)),
        ), (1, 1, 1))
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        .batch(config['batch_size'])
    )
