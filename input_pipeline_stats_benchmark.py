import tempfile
import os
import tensorflow as tf
from functools import partial
from tensorflow import keras
import numpy as np
from tensorflow import TensorSpec

print(tf.__version__)

name='benchmark'

path = os.path.join(tempfile.gettempdir() + "/data/"+name+"/data/", "experiment_saved_data")
print("base path: " + path)
logs_path = 'logs'
debug_dir = 'debug'

debug_verbose = False
debug_dump = False

is_dist_dataset = True

data_size = 24 if debug_verbose else 100000
batch_size = 4 if debug_verbose else 100
steps_per_epoch = int(round(data_size / batch_size))
num_shards = 4
num_workers = 1
epochs = 2

feature_group_dim = 3 if debug_verbose else 100
feature_groups_dim = 2 if debug_verbose else 3

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
#Forcing to use specific number of GPUs
tf.config.experimental.set_visible_devices(physical_devices_gpu[0:num_workers], 'GPU')

tf.config.threading.set_inter_op_parallelism_threads(num_workers)
tf.config.threading.set_intra_op_parallelism_threads(num_workers)

logical_devices_cpu = tf.config.list_logical_devices('CPU')
print("CPU Logical After Config: " + str(logical_devices_cpu))

print("GPU Physical After Config: " + str(physical_devices_gpu))
logical_devices_gpu = tf.config.list_logical_devices('GPU')
print("GPU Logical: " + str(logical_devices_gpu))

print("TF Config: " + str(tf.config))

if debug_dump:
    tf.debugging.experimental.enable_dump_debug_info(
        dump_root=path + "/" + logs_path+"/" + debug_dir,
        tensor_debug_mode="FULL_HEALTH",
        circular_buffer_size=-1, op_regex=None, tensor_dtypes=None)


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


    #prepare the metadata:
    feature_mappings = {str(f): feature_group_dim for f in range(feature_groups_dim)}
    print(feature_mappings)

    feature_groups_specs = (tuple(tf.TensorSpec(shape=(None, feature_group_dim), dtype=tf.float32, name=None) for f in range(feature_groups_dim)))

    tensor_spec = (TensorSpec(shape=(), dtype=tf.int64, name=None),#enum for sharding, to be removed after loading the data
                   # Keras Input: (inputs, targets, sample_weights)
                   (feature_groups_specs,#Feature groups, inputs
                    TensorSpec(shape=(None, 1), dtype=tf.float32, name=None),#labels, targets
                    TensorSpec(shape=(None, 1), dtype=tf.float32, name=None)))#weights, sample_weights
    print(tensor_spec)

    feature_groups_types = (tuple(tf.float32 for f in range(feature_groups_dim)))
    print(feature_groups_types)

    numbers = range(feature_group_dim)
    sequence_of_numbers = [number for number in numbers]

    if debug_verbose:
        print(sequence_of_numbers)

    #random - not used
    # feature_groups = (tuple(np.random.uniform(0, 1, feature_group_dim) for f in range(feature_groups_dim)))
    #sequential - to test for immutability within each feature group
    feature_groups = (tuple(sequence_of_numbers for f in range(feature_groups_dim)))
    #labels are random numbers, weights are sequential to test for the shuffling behavior:
    elements = [(feature_groups,np.random.uniform(0, 1, 1), [n]) for n in range(data_size)]

    dataset = tf.data.Dataset.from_generator(
        lambda: iter(elements), (feature_groups_types, tf.float32, tf.float32))
    if debug_verbose:
        print("initial ds"+"\n")
        for elem in dataset:
          print("\t"+str(elem))
        print("=============="+"\n\n")


    dataset = dataset.batch(batch_size)
    if debug_verbose:
        print("batched ds, added arrays"+"\n")
        for elem in dataset:
          print("\t"+str(elem))
        print("=============="+"\n\n")

    dataset = dataset.enumerate()
    if debug_verbose:
        print("enumed ds, added a new element for sharding strategy"+"\n")
        for elem in dataset:
          print("\t"+str(elem))
        print("=============="+"\n\n")

    tf.data.experimental.save(dataset, path, shard_func=lambda x, y: x % num_shards)

    new_dataset = tf.data.experimental.load(path, tensor_spec, reader_func=partial_reader_func)
    if debug_verbose:
        print("print ds after loading - notice the shuffled batches"+"\n")
        for elem in new_dataset:
          print("\t"+str(elem))
        print("=============="+"\n\n")

    new_dataset = new_dataset.map(lambda x, y: y)
    if debug_verbose:
        print("print ds after loading and de-enuming - back to the original elements, no id"+"\n")
        for elem in new_dataset:
          print("\t"+str(elem))
        print("=============="+"\n\n")

    new_dataset = new_dataset.repeat(epochs)
    if debug_verbose:
        print("print ds after applying repeat and de-enuming - back to the original elements, no id" + "\n")
        for elem in new_dataset:
            print("\t" + str(elem))
        print("==============" + "\n\n")

    options = tf.data.Options()
    if is_dist_dataset:
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
        new_dataset = new_dataset.with_options(options)
        new_dataset = strategy.experimental_distribute_dataset(new_dataset)
    else:
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        new_dataset = new_dataset.with_options(options)

    if debug_verbose:
        for epoch in range(epochs):
            print("epoch " + str(epoch) + ": print dist ds - per replica batches are already shuffled and sharded, notice the batch size reduction proportional to the number of workers"+"\n")
            for elem in new_dataset:
              print("\t"+str(elem))
            print("==============" + "\n\n")

    inputs = [keras.Input(shape=(dim,), name=name) for name, dim in feature_mappings.items()]

    m = keras.layers.concatenate(inputs) if len(inputs) > 1 else inputs[0]

    x = keras.layers.Dropout(0.1, seed=1)(m)
    x = keras.layers.Dense(128, 'relu')(x)
    x = keras.layers.Dense(1, activation='sigmoid',
                           kernel_regularizer=keras.regularizers.l2(0.0001),
                           name='output')(x)
    m = keras.Model(inputs=inputs, outputs=x, name=name)
    print(m.summary())

    model_optimizer = keras.optimizers.Adamax()
    m.compile(optimizer=model_optimizer,
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
            profile_batch=(batch_size*10 + 1, batch_size*10 + 2))
    ]

    history = m.fit(new_dataset,
                        epochs=epochs,
                        validation_data=new_dataset,
                        validation_freq=1,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=steps_per_epoch,
                        max_queue_size=batch_size,
                        callbacks=callbacks,
                        workers=1,
                        use_multiprocessing=False)

    print('Training finished.')
    print(history.history)
