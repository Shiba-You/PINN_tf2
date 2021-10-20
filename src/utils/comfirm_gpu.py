import tensorflow as tf

def comfirm_gpu(gpu_flg):
    gpus = tf.config.experimental.list_physical_devices("GPU")

    if gpu_flg == 0:
        print("gpu_flg:", gpu_flg)
        if gpus:
            # Restrict TensorFlow to only use the first GPU
            try:
                tf.config.experimental.set_visible_devices(gpus[0], "GPU")
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                print(e)
        
    elif gpu_flg == 1:
        print("gpu_flg:", gpu_flg)
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
        
    elif gpu_flg == 2:
        print("gpu_flg:", gpu_flg)
        if gpus:
            # Create 2 virtual GPUs with 1GB memory each
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 2 ** 10),
                    tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 2 ** 10)])
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)
                
    elif gpu_flg == 4:
        print("gpu_flg:", gpu_flg)
        if gpus:
            # Create 4 virtual GPUs with 1GB memory each
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 2 ** 10),
                    tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 2 ** 10),
                    tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 2 ** 10),
                    tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 2 ** 10)])
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)

    else:
        print(">>>>> ERROR: gpu_flg not specified properly")