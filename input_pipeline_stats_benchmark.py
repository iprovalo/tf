import tempfile
import os
import tensorflow as tf
from functools import partial
from tensorflow import keras
from tensorflow.python.keras.optimizer_v2 import adam


name='benchmark'

path = os.path.join(tempfile.gettempdir() + "/data/"+name+"/data/", "experiment_saved_data")
print("base path: " + path)
logs_path = 'logs'

data_size = 100000
batch_size = 10
steps_per_epoch = int(round(data_size / batch_size))
num_shards = 10
num_workers = 1
epochs = 2


# tf.debugging.set_log_device_placement(True)
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['NCCL_DEBUG'] = 'INFO'

physical_devices_cpu = tf.config.list_physical_devices('CPU')
print("CPU Physical: " + str(physical_devices_cpu))

tf.config.set_logical_device_configuration(physical_devices_cpu[0],
                                           [tf.config.LogicalDeviceConfiguration() for number in
                                            range(num_workers)])

physical_devices_gpu = tf.config.list_physical_devices('GPU')
print("GPU Physical: " + str(physical_devices_gpu))
tf.config.experimental.set_visible_devices(physical_devices_gpu[0:num_workers], 'GPU') #TODO: temp forcing to use a single GPU

tf.config.threading.set_inter_op_parallelism_threads(num_workers)
tf.config.threading.set_intra_op_parallelism_threads(num_workers)

logical_devices_cpu = tf.config.list_logical_devices('CPU')
print("CPU Logical After Config: " + str(logical_devices_cpu))

print("GPU Physical After Config: " + str(physical_devices_gpu))
logical_devices_gpu = tf.config.list_logical_devices('GPU')
print("GPU Logical: " + str(logical_devices_gpu))

print("GPU Device: " + tf.test.gpu_device_name())
print("TF Config: " + str(tf.config))



# tf.debugging.experimental.enable_dump_debug_info(
#     dump_root=path + "/" + logs_path,
#     tensor_debug_mode="FULL_HEALTH",
#     circular_buffer_size=-1, op_regex=None, tensor_dtypes=None)


strategy = tf.distribute.MirroredStrategy()

with strategy.scope():

    def custom_reader_func(dataset, num_shards=1):
        dataset = dataset.shuffle(num_shards)
        dataset = dataset.interleave(
            lambda x: x,
            cycle_length=tf.data.experimental.AUTOTUNE,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=False)
        return dataset.prefetch(tf.data.experimental.AUTOTUNE)

    partial_reader_func = partial(custom_reader_func, num_shards=num_shards)


    dataset = tf.data.Dataset.range(data_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.enumerate()

    tf.data.experimental.save(dataset, path, shard_func=lambda x, y: x % num_shards)

    new_dataset = tf.data.experimental.load(path, (tf.TensorSpec(shape=(), dtype=tf.int64, name=None), tf.TensorSpec(shape=(batch_size,), dtype=tf.int64)), reader_func=partial_reader_func)
    new_dataset = new_dataset.map(lambda x, y: y)
    new_dataset = new_dataset.repeat(epochs)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    new_dataset = new_dataset.with_options(options)
    dist_dataset = strategy.experimental_distribute_dataset(new_dataset)

    inputs = [keras.Input(shape=(1,))]

    m = keras.layers.concatenate(inputs) if len(inputs) > 1 else inputs[0]

    x = keras.layers.Dropout(0.1, seed=1)(m)
    x = keras.layers.Dense(128, 'relu')(x)
    x = keras.layers.Dense(1, activation='sigmoid',
                           kernel_regularizer=keras.regularizers.l2(0.0001),
                           name='output')(x)
    m = keras.Model(inputs=inputs, outputs=x, name=name)
    print(m.summary())

    m.compile(optimizer=adam.Adam(learning_rate=0.001),
                  loss=keras.losses.binary_crossentropy,
                  metrics=[keras.metrics.BinaryAccuracy()])

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(path, logs_path, m.name + '_{epoch}.ckpt'),
            save_weights_only=True,
            save_best_only=False,
            monitor='val_loss',
            verbose=True),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(path, logs_path),
            update_freq='epoch',
            histogram_freq=1,  # epochs before logging weight histogram with val data
            profile_batch=(batch_size + 1, batch_size + 2))
    ]

    history = m.fit(dist_dataset,
                        epochs=epochs,
                        validation_data=dist_dataset,
                        validation_freq=1,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=steps_per_epoch,
                        max_queue_size=batch_size,
                        callbacks=callbacks,
                        workers=1,
                        use_multiprocessing=False)

    print('Training finished.')
    print(history.history)

