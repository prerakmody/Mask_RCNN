"""
Mask R-CNN
Multi-GPU Support for Keras.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

Ideas and a small code snippets from these sources:
https://github.com/fchollet/keras/issues/2436
https://medium.com/@kuza55/transparent-multi-gpu-training-on-tensorflow-with-keras-8b0016fd9012
https://github.com/avolkov1/keras_experiments/blob/master/keras_exp/multigpu/
https://github.com/fchollet/keras/blob/master/keras/utils/training_utils.py
"""

import tensorflow as tf
import keras.backend as K
import keras.layers as KL
import keras.models as KM


def multi_gpu_model(model, gpus=None):
    """Replicates a model on different GPUs.
    Specifically, this function implements single-machine
    multi-GPU data parallelism. It works in the following way:
    - Divide the model's input(s) into multiple sub-batches.
    - Apply a model copy on each sub-batch. Every model copy
        is executed on a dedicated GPU.
    - Concatenate the results (on CPU) into one big batch.
    E.g. if your `batch_size` is 64 and you use `gpus=2`,
    then we will divide the input into 2 sub-batches of 32 samples,
    process each sub-batch on one GPU, then return the full
    batch of 64 processed samples.
    This induces quasi-linear speedup on up to 8 GPUs.
    This function is only available with the TensorFlow backend
    for the time being.
    # Arguments
        model: A Keras model instance. To avoid OOM errors,
            this model could have been built on CPU, for instance
            (see usage example below).
        gpus: Integer >= 2 or list of integers, number of GPUs or
            list of GPU IDs on which to create model replicas.
    # Returns
        A Keras `Model` instance which can be used just like the initial
        `model` argument, but which distributes its workload on multiple GPUs.
    # Example
    ```python
        import tensorflow as tf
        from keras.applications import Xception
        from keras.utils import multi_gpu_model
        import numpy as np
        num_samples = 1000
        height = 224
        width = 224
        num_classes = 1000
        # Instantiate the base model (or "template" model).
        # We recommend doing this with under a CPU device scope,
        # so that the model's weights are hosted on CPU memory.
        # Otherwise they may end up hosted on a GPU, which would
        # complicate weight sharing.
        with tf.device('/cpu:0'):
            model = Xception(weights=None,
                             input_shape=(height, width, 3),
                             classes=num_classes)
        # Replicates the model on 8 GPUs.
        # This assumes that your machine has 8 available GPUs.
        parallel_model = multi_gpu_model(model, gpus=8)
        parallel_model.compile(loss='categorical_crossentropy',
                               optimizer='rmsprop')
        # Generate dummy data.
        x = np.random.random((num_samples, height, width, 3))
        y = np.random.random((num_samples, num_classes))
        # This `fit` call will be distributed on 8 GPUs.
        # Since the batch size is 256, each GPU will process 32 samples.
        parallel_model.fit(x, y, epochs=20, batch_size=256)
        # Save model via the template model (which shares the same weights):
        model.save('my_model.h5')
    ```
    # On model saving
    To save the multi-gpu model, use `.save(fname)` or `.save_weights(fname)`
    with the template model (the argument you passed to `multi_gpu_model`),
    rather than the model returned by `multi_gpu_model`.
    """
    """
    if K.backend() != 'tensorflow':
        raise ValueError('`multi_gpu_model` is only available '
                         'with the TensorFlow backend.')

    available_devices = _get_available_devices()
    available_devices = [_normalize_device_name(name) for name in available_devices]
    if not gpus:
        # Using all visible GPUs when not specifying `gpus`
        # e.g. CUDA_VISIBLE_DEVICES=0,2 python3 keras_mgpu.py
        gpus = len([x for x in available_devices if 'gpu' in x])
    """
    
    if isinstance(gpus, (list, tuple)):
        if len(gpus) <= 1:
            raise ValueError('For multi-gpu usage to be effective, '
                             'call `multi_gpu_model` with `len(gpus) >= 2`. '
                             'Received: `gpus=%s`' % gpus)
        num_gpus = len(gpus)
        target_gpu_ids = gpus
    else:
        if gpus <= 1:
            raise ValueError('For multi-gpu usage to be effective, '
                             'call `multi_gpu_model` with `gpus >= 2`. '
                             'Received: `gpus=%d`' % gpus)
        num_gpus = gpus
        target_gpu_ids = range(num_gpus)

    import tensorflow as tf

    target_devices = ['/cpu:0'] + ['/gpu:%d' % i for i in target_gpu_ids]
    
    def get_slice(data, i, parts):
        shape       = tf.shape(data)
        batch_size  = shape[:1]
        input_shape = shape[1:]
        step = batch_size // parts
        
        if i == num_gpus - 1:
            size = batch_size - step * i
        else:
            size = step
        
        size = tf.concat([size, input_shape], axis=0)
        stride = tf.concat([step, input_shape * 0], axis=0)
        start = stride * i
        
        return tf.slice(data, start, size)

    all_outputs = []
    for i in range(len(model.outputs)):
        all_outputs.append([])

    # Place a copy of the model on each GPU,
    # each getting a slice of the inputs.
    for i, gpu_id in enumerate(target_gpu_ids):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('replica_%d' % gpu_id):
                inputs = []
                # Retrieve a slice of the input.
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_i = KL.Lambda(get_slice,
                                     output_shape=input_shape,
                                     arguments={'i': i,
                                                'parts': num_gpus})(x)
                    inputs.append(slice_i)

                # Apply model on slice
                # (creating a model replica on the target device).
                outputs = model(inputs)
                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save the outputs for merging back together later.
                for o in range(len(outputs)):
                    all_outputs[o].append(outputs[o])

    # Merge outputs on CPU.
    with tf.device('/cpu:0'):
        merged = []
        for name, outputs in zip(model.output_names, all_outputs):
            # If outputs are numbers without dimensions, add a batch dim.
            def add_dim(tensor):
                """Add a dimension to tensors that don't have any."""
                if K.int_shape(tensor) == ():
                    return KL.Lambda(lambda t: K.reshape(t, [1, 1]))(tensor)
                return tensor
            outputs = list(map(add_dim, outputs))
            
            verbose = 0
            if verbose:
                print ('---------------->')
                for each in outputs:
                    print (each)
                    
            merged.append(KL.concatenate(outputs,
                                      axis=0, name=name))
        return KM.Model(model.inputs, merged)

class ParallelModel(KM.Model):
    """Subclasses the standard Keras Model and adds multi-GPU support.
    It works by creating a copy of the model on each GPU. Then it slices
    the inputs and sends a slice to each copy of the model, and then
    merges the outputs together and applies the loss on the combined
    outputs.
    """

    def __init__(self, keras_model, gpu_count, verbose = 0):
        """Class constructor.
        keras_model: The Keras model to parallelize
        gpu_count: Number of GPUs. Must be > 1
        """
        self.verbose = verbose    
        self.inner_model = keras_model
        self.gpu_count = gpu_count
        merged_outputs = self.make_parallel()
        super(ParallelModel, self).__init__(inputs=self.inner_model.inputs,
                                            outputs=merged_outputs)

    def __getattribute__(self, attrname):
        """Redirect loading and saving methods to the inner model. That's where
        the weights are stored."""
        if 'load' in attrname or 'save' in attrname:
            return getattr(self.inner_model, attrname)
        return super(ParallelModel, self).__getattribute__(attrname)

    def summary(self, *args, **kwargs):
        """Override summary() to display summaries of both, the wrapper
        and inner models."""
        super(ParallelModel, self).summary(*args, **kwargs)
        self.inner_model.summary(*args, **kwargs)

    def make_parallel(self):
        """Creates a new wrapper model that consists of multiple replicas of
        the original model placed on different GPUs.
        """
        # Slice inputs. Slice inputs on the CPU to avoid sending a copy
        # of the full inputs to all GPUs. Saves on bandwidth and memory.
        if self.verbose:
            for each in zip(self.inner_model.input_names, self.inner_model.inputs):
                print ('---> ', each)
        input_slices = {name: tf.split(x, self.gpu_count)
                        for name, x in zip(self.inner_model.input_names,
                                           self.inner_model.inputs)}

        output_names = self.inner_model.output_names
        outputs_all = []
        for i in range(len(self.inner_model.outputs)):
            outputs_all.append([])

        # Run the model call() on each GPU to place the ops there
        for i in range(self.gpu_count):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('tower_%d' % i):
                    # Run a slice of inputs through this replica
                    zipped_inputs = zip(self.inner_model.input_names,
                                        self.inner_model.inputs)
                    inputs = [
                        KL.Lambda(lambda s: input_slices[name][i],
                                  output_shape=lambda s: (None,) + s[1:])(tensor)
                        for name, tensor in zipped_inputs]
                    # Create the model replica and get the outputs
                    if self.verbose:
                        if i == 0:
                            print ('\ntower_{0} - i/p '.format(i))
                            for each in inputs:
                                print ('--->', each)
                    
                    outputs = self.inner_model(inputs)
                    if self.verbose:
                        if i == 0:
                            print ('\ntower_{0} - o/p '.format(i))
                            for each in outputs:
                                print ('--->', each)
                                
                    if not isinstance(outputs, list):
                        outputs = [outputs]
                    # Save the outputs for merging back together later
                    for l, o in enumerate(outputs):
                        outputs_all[l].append(o)

        # Merge outputs on CPU
        with tf.device('/cpu:0'):
            merged = []
            for outputs, name in zip(outputs_all, output_names):
                # If outputs are numbers without dimensions, add a batch dim.
                def add_dim(tensor):
                    """Add a dimension to tensors that don't have any."""
                    if K.int_shape(tensor) == ():
                        return KL.Lambda(lambda t: K.reshape(t, [1, 1]))(tensor)
                    return tensor
                outputs = list(map(add_dim, outputs))
                
                verbose = 0
                if verbose:
                    print ('---------------->')
                    for each in outputs:
                        print (each)
                
                # Concatenate
                merged.append(KL.Concatenate(axis=0, name=name)(outputs))
        return merged


# if __name__ == "__main__":
    # Testing code below. It creates a simple model to train on MNIST and
    # tries to run it on 2 GPUs. It saves the graph so it can be viewed
    # in TensorBoard. Run it as:
    #
    # python3 parallel_model.py

#     import os
#     import numpy as np
#     import keras.optimizers
#     from keras.datasets import mnist
#     from keras.preprocessing.image import ImageDataGenerator

#     GPU_COUNT = 2

#     # Root directory of the project
#     ROOT_DIR = os.getcwd()

#     # Directory to save logs and trained model
#     MODEL_DIR = os.path.join(ROOT_DIR, "logs/parallel")

#     def build_model(x_train, num_classes):
#         # Reset default graph. Keras leaves old ops in the graph,
#         # which are ignored for execution but clutter graph
#         # visualization in TensorBoard.
#         tf.reset_default_graph()

#         inputs = KL.Input(shape=x_train.shape[1:], name="input_image")
#         x = KL.Conv2D(32, (3, 3), activation='relu', padding="same",
#                       name="conv1")(inputs)
#         x = KL.Conv2D(64, (3, 3), activation='relu', padding="same",
#                       name="conv2")(x)
#         x = KL.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
#         x = KL.Flatten(name="flat1")(x)
#         x = KL.Dense(128, activation='relu', name="dense1")(x)
#         x = KL.Dense(num_classes, activation='softmax', name="dense2")(x)

#         return KM.Model(inputs, x, "digit_classifier_model")

#     # Load MNIST Data
#     (x_train, y_train), (x_test, y_test) = mnist.load_data()
#     x_train = np.expand_dims(x_train, -1).astype('float32') / 255
#     x_test = np.expand_dims(x_test, -1).astype('float32') / 255

#     print('x_train shape:', x_train.shape)
#     print('x_test shape:', x_test.shape)

#     # Build data generator and model
#     datagen = ImageDataGenerator()
#     model = build_model(x_train, 10)

#     # Add multi-GPU support.
#     model = ParallelModel(model, GPU_COUNT)

#     optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=5.0)

#     model.compile(loss='sparse_categorical_crossentropy',
#                   optimizer=optimizer, metrics=['accuracy'])

#     model.summary()

#     # Train
#     model.fit_generator(
#         datagen.flow(x_train, y_train, batch_size=64),
#         steps_per_epoch=50, epochs=10, verbose=1,
#         validation_data=(x_test, y_test),
#         callbacks=[keras.callbacks.TensorBoard(log_dir=MODEL_DIR,
#                                                write_graph=True)]
#     )
