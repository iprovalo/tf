import tempfile
import os
import tensorflow as tf
from functools import partial
from tensorflow import keras
from tensorflow.python.keras.optimizer_v2 import adam

path = os.path.join(tempfile.gettempdir(), "saved_data")

name='benchmark'
data_size = 1000
batch_size = 20
num_shards = 2
num_workers = 2
epochs = 30

physical_devices_cpu = tf.config.list_physical_devices('CPU')
print("CPU Physical: " + str(physical_devices_cpu))
tf.config.set_logical_device_configuration(physical_devices_cpu[0], [tf.config.LogicalDeviceConfiguration() for number in range(num_workers)])
logical_devices_cpu = tf.config.list_logical_devices('CPU')
print("CPU Logical After Config: " + str(logical_devices_cpu))


def custom_reader_func(dataset, num_shards=1):
    dataset = dataset.shuffle(num_shards)
    dataset = dataset.interleave(
        lambda x: x,
        cycle_length=tf.data.experimental.AUTOTUNE,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=False)
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)

partial_reader_func = partial(custom_reader_func, num_shards=num_shards)


# Save a dataset
dataset = tf.data.Dataset.range(data_size)
print("initial ds"+"\n")
for elem in dataset:
  print("\t"+str(elem))
print("=============="+"\n\n")

dataset = dataset.batch(batch_size)
print("batched ds, added arrays"+"\n")
for elem in dataset:
  print("\t"+str(elem))
print("=============="+"\n\n")

dataset = dataset.enumerate()
print("enumed ds, added a new element for sharding strategy"+"\n")
for elem in dataset:
  print("\t"+str(elem))
print("=============="+"\n\n")

tf.data.experimental.save(dataset, path, shard_func=lambda x, y: x % num_shards)

print("ds after saving - should be no change"+"\n")
for elem in dataset:
  print("\t"+str(elem))
print("=============="+"\n\n")

new_dataset = tf.data.experimental.load(path, (tf.TensorSpec(shape=(), dtype=tf.int64, name=None), tf.TensorSpec(shape=(batch_size,), dtype=tf.int64)), reader_func=partial_reader_func)
print("print ds after loading - notice the shuffled batches"+"\n")
for elem in new_dataset:
  print("\t"+str(elem))
print("=============="+"\n\n")

new_dataset = new_dataset.map(lambda x, y: y)
print("print ds after loading and de-enuming - back to the original elements, no id"+"\n")
for elem in new_dataset:
  print("\t"+str(elem))
print("=============="+"\n\n")

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
    dataset = dataset.with_options(options)
    dist_dataset = strategy.experimental_distribute_dataset(new_dataset)

    for epoch in range(epochs):
        print("epoch " + str(epoch) + ": print dist ds - per replica batches are already shuffled and sharded, notice the batch size reduction proportional to the number of workers"+"\n")
        for elem in dist_dataset:
          print("\t"+str(elem))
        print("==============" + "\n\n")


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
            filepath=os.path.join(path, 'logs', m.name + '_{epoch}.ckpt'),
            save_weights_only=True,
            save_best_only=False,
            monitor='val_loss',
            verbose=True)
    ]

    history = m.fit(dist_dataset,
                        epochs=epochs,
                        validation_data=dist_dataset,
                        validation_freq=1,
                        steps_per_epoch=int(round(data_size / batch_size / num_shards)),
                        validation_steps=int(round(data_size / batch_size / num_shards)),
                        max_queue_size=batch_size,
                        callbacks=callbacks,
                        workers=1,
                        use_multiprocessing=False)

    print('Training finished.')
    print(history.history)
