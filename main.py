import itertools
import math
import pathlib
import random
import sys
import getopt
import time


import matplotlib.pyplot as plt
import hashlib
import scipy
import scipy.signal
import tensorflow as tf
import keras
import keras.layers
import keras.losses
import keras.optimizers
import keras.callbacks
import keras.mixed_precision
import dct4stream
import dct4stream2
import re
import numpy as np
import os

FFT_SIZE = 256
BLOCK_SIZE = 256

def dB(val):
    return math.pow(10.0, val/20.0)


def plot_block(blk, title, fignum=None, show=True):
    if fignum is not None:
        plt.figure(fignum)
    # plt.pcolormesh(np.abs(blk).T, vmin=0, vmax=1)
    plt.pcolormesh(np.abs(blk).T, norm='log', vmin=dB(-120), vmax=1)
    plt.colorbar()
    plt.title(title)
    if show:
        plt.show()


def ideal_training_noise_block(intensity_dB = -50, length = 256):
    sigma = dB(intensity_dB)
    return np.array([
        scipy.fftpack.dct(column*scipy.signal.windows.blackmanharris(column.shape[0]), type=4, norm='ortho')
        for column in np.random.normal(0.0, sigma, (length,256)).astype(np.float32)
    ])


def preprocess_dataset(dataset_dir, output_dir):
    sampleId = 0
    os.makedirs(output_dir, exist_ok=True)

    files = [
        os.path.join(dataset_dir, filename)
        for filename in os.listdir(dataset_dir)
        if os.path.splitext(filename)[1] == ".wav"
    ]

    for file in files:
        print("processing: {}...".format(file), end="", flush=True)
        for blockIdx, block in enumerate(dct4stream2.AudioInputDCT4(file)):
            progress = " [blk {}]".format(blockIdx)
            n_rewind = len(progress)
            print(progress, end="", flush=True)
            for chIdx, chBlkData in enumerate([block[:, :, 0], block[:, :, 1]]):
                sigma = dB(random.uniform(-50.5, -49.5))
                noise = np.array([
                    scipy.fftpack.dct(column*scipy.signal.windows.blackmanharris(column.shape[0]), type=4, norm='ortho')
                    for column in np.random.normal(0.0, sigma, chBlkData.shape).astype(np.float32)
                ])
                gain = random.uniform(-50, 10)
                noisy_signal = chBlkData*dB(gain)+noise


                # plot_block(noisy_signal, "sig+noise (gain={} dB)".format(gain))
                # plot_block(noise, "noise (gain={} dB)".format(gain))

                np.save(os.path.join(output_dir, "sample-{:06d}.npy".format(sampleId)), noisy_signal.astype(np.float16))
                np.save(os.path.join(output_dir, "noise-{:06d}.npy".format(sampleId)), noise.astype(np.float16))

                sampleId += 1
            print("\b" * n_rewind, end="", flush=True)
        print("", flush=True)


def str_to_probability(in_str):
    seed = in_str.encode()
    hash_digest = hashlib.sha512(seed).digest()
    hash_int = int.from_bytes(hash_digest, 'big')
    return hash_int / (2**(hashlib.sha512().digest_size * 8))


def preproc_dataset_entry():
    input_dir = "."
    output_dir = "outputs"
    opts, args = getopt.getopt(sys.argv[1:], "i:o:", ["idir=", "odir="])
    for opt, arg in opts:
        if opt in ("-i", "--idir"):
            input_dir = arg
        elif opt in ("-o", "--odir"):
            output_dir = arg
    print("input={}, output={}".format(input_dir, output_dir))
    preprocess_dataset(input_dir, output_dir)


def first_number_substr(text: str) -> str:
    return re.search(r'\d+', text).group()


def path_to_sampleid(path) -> str:
    return first_number_substr(path.split("/")[-1])


COMPANDING_SCALE_DB = 120.0


def compress(arr):
    intensity_db = 20.0 * np.log10(np.clip(np.abs(arr), a_min=dB(-COMPANDING_SCALE_DB), a_max=None))
    intensity_norm = intensity_db / COMPANDING_SCALE_DB + 1.0
    return intensity_norm * np.sign(arr)


def expand(arr):
    intensity_db = (np.abs(arr) - 1.0) * COMPANDING_SCALE_DB
    intensity_real = np.power(10.0, intensity_db / 20.0)
    return intensity_real * np.sign(arr)


def load_sample_mapfn(noisy_filename, residual_filename):
    if path_to_sampleid(noisy_filename) != path_to_sampleid(residual_filename):
        print("!!MISMATCH!! {} =/= {}".format(noisy_filename, residual_filename))
    noisy = np.load(noisy_filename).astype(np.float32)
    residual = np.load(residual_filename).astype(np.float32)
    noisy.resize((BLOCK_SIZE, FFT_SIZE, 1))
    residual.resize((BLOCK_SIZE, FFT_SIZE, 1))

    c_noisy = np.abs(compress(noisy))
    # c_residual = np.abs(compress(residual))
    c_clean = np.abs(compress(noisy-residual))

    return (c_noisy).astype(np.float32), (c_noisy-c_clean).astype(np.float32)


def sample_generator(noisy_filenames, residual_filenames):
    num_samples = len(noisy_filenames)
    for i in random.sample(range(num_samples), k=num_samples):
        arr = load_sample_mapfn(noisy_filenames[i], residual_filenames[i])
        yield arr


def gen_dataset_subset(noisy_filenames, residual_filenames, k):
    num_samples = len(noisy_filenames)
    inputs = []
    outputs = []
    for i in random.sample(range(num_samples), k=k):
        arr = load_sample_mapfn(noisy_filenames[i], residual_filenames[i])
        inputs.append(arr[0])
        outputs.append(arr[1])
    return np.array(inputs), np.array(outputs)


def get_dataset_filenames(paths, ds_type, side):
    outputs = []
    for path in paths:
        subset = "train"
        x = str_to_probability(path_to_sampleid(path))
        if 0 <= x < 0.05:
            subset = "test"
        elif 0.05 <= x < 0.1:
            subset = "val"
        elif 0.1 <= x:
            subset = "train"

        if ds_type == subset and side in path:
            outputs.append(path)

    return outputs


def dncnn_audio(depth, flt=64):
    input_layer = keras.layers.Input(shape=(None, FFT_SIZE, 1))
    # x = keras.layers.Rescaling(scale=0.5, offset=0.5)(input_layer)
    x = keras.layers.Conv2D(filters=flt, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    x = keras.layers.ReLU()(x)
    for i in range(depth):
        x = keras.layers.Conv2D(filters=flt, kernel_size=(3, 5), strides=(1, 1), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    # x = keras.layers.Rescaling(scale=2.0, offset=-1.0)(x)
    # x = keras.layers.Subtract()([input_layer, x])
    return keras.Model(inputs=input_layer, outputs=x)


def stupid_model(depth):
    input_layer = keras.layers.Input(shape=(None, FFT_SIZE))
    x = keras.layers.LSTM(units=FFT_SIZE, return_sequences=True)(input_layer)
    x = keras.layers.Concatenate()([x, input_layer])
    x = keras.layers.Dense(FFT_SIZE*2)(x)
    x = keras.layers.ReLU()(x)
    for i in range(depth):
        x = keras.layers.Dense(FFT_SIZE)(x)
        x = keras.layers.ReLU()(x)
    x = keras.layers.Dense(FFT_SIZE)(x)
    return keras.Model(inputs=input_layer, outputs=x)


def lr_scheduler(epoch, lr):
    if epoch < 16:
        return 0.001
    else:
        return 0.001 / math.log2(epoch / 8)


def train(dataset_dir):

    print("scanning dataset...")
    ds_paths = [os.path.join(dataset_dir, filename)
                for filename in os.listdir(dataset_dir)
                if ".npy" in filename]
    ds_paths.sort()
    print("done")
    #
    # policy = keras.mixed_precision.Policy('mixed_float16')
    # keras.mixed_precision.set_global_policy(policy)

    paths_in_train = get_dataset_filenames(ds_paths, "train", "sample-")
    paths_in_val = get_dataset_filenames(ds_paths, "val", "sample-")
    paths_in_test = get_dataset_filenames(ds_paths, "test", "sample-")

    paths_out_train = get_dataset_filenames(ds_paths, "train", "noise-")
    paths_out_val = get_dataset_filenames(ds_paths, "val", "noise-")
    paths_out_test = get_dataset_filenames(ds_paths, "test", "noise-")

    blk_shape = (BLOCK_SIZE, FFT_SIZE)

    blk1 = 0.5 * compress(np.load(paths_in_train[402])) + 0.5
    blk2 = 0.5 * compress(np.load(paths_out_train[402])) + 0.5
    plt.figure()
    plot_block(blk1, "noisy signal", show=False)
    plt.figure()
    plot_block(blk2, "noise only")

    batch_size = 8

    dataset_train = tf.data.Dataset\
        .from_generator(
            lambda: sample_generator(paths_in_train, paths_out_train),
            output_signature=(
                tf.TensorSpec(shape=(None, FFT_SIZE, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, FFT_SIZE, 1), dtype=tf.float32)
            )
        ) \
        .prefetch(tf.data.AUTOTUNE) \
        .batch(batch_size)

    dataset_val = tf.data.Dataset \
        .from_generator(
            lambda: sample_generator(paths_in_val, paths_out_val),
            output_signature=(
                tf.TensorSpec(shape=(None, FFT_SIZE, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, FFT_SIZE, 1), dtype=tf.float32)
            )
        ) \
        .prefetch(tf.data.AUTOTUNE) \
        .batch(batch_size)

    # model = dncnn_audio(16, 48)
    model = keras.models.load_model("nrcnn.base.keras")
    # model = stupid_model(10)

    model.summary()
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(learning_rate=0.0008)
    )
    model.save("init.keras")

    savecb = keras.callbacks.ModelCheckpoint(
        "saved-model-epoch{epoch:03d}-{val_loss:.10f}.keras",
        monitor='val_loss',
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode='min'
    )

    lrsched = keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

    # dataset_train_tmp = gen_dataset_subset(paths_in_train, paths_out_train, 3200)
    # dataset_val_tmp = gen_dataset_subset(paths_in_val, paths_out_val, 100)

    # history = model.fit(x=dataset_train_tmp[0], y=dataset_train_tmp[1], batch_size=1, epochs=60, validation_data=dataset_val_tmp, callbacks=savecb)

    history = model.fit(dataset_train, validation_data=dataset_val, epochs=60, callbacks=[savecb])
    model.save("final.keras")



def denoise_blocks(model, blocks: np.ndarray): # blocks: batch-chunk-bin-channel

    batch_size = blocks.shape[0]
    num_channels = blocks.shape[3]
    out_blocks = []
    t0 = time.time()
    # denoised_block = block.copy() # batch, chunk, bin, channel
    exp_blocks = np.expand_dims(blocks, axis=-1) # batch, chunk, bin, channel, tf-channel
    transposed_blocks = np.transpose(exp_blocks, axes=(0, 3, 1, 2, 4))
    cnn_input = np.abs(compress(np.concatenate(transposed_blocks))).copy() # batch+channel, chunk, bin, tf-channel
    t1 = time.time()
    predicted_noise_comp_batched = model.predict(cnn_input) # batch+channel, chunk, bin, tf-channel
    t2 = time.time()
    # plot_block(cnn_input[0,:,:,0], "before cnn / {}".format(cnn_input.shape), fignum=150, show=False)
    # plot_block(predicted_noise_comp_batched[0,:,:,0], "after cnn / {}".format(predicted_noise_comp_batched.shape), fignum=151)

    for batch_idx in range(batch_size):
        predicted_noise_comp = np.squeeze(np.array(predicted_noise_comp_batched[batch_idx*num_channels:(batch_idx+1)*num_channels])) # channel-chunk-bin
        predicted_noise_comp = np.transpose(predicted_noise_comp, axes=(1,2,0)) # chunk-bin-channel
        source_comp = np.abs(compress(blocks[batch_idx]))
        predicted_clean_comp = source_comp - predicted_noise_comp
        predicted_clean = expand(predicted_clean_comp) * np.sign(blocks[batch_idx])
        out_blocks.append(predicted_clean)

    t3 = time.time()

    # print("prep took {} ms, inference took {} ms".format(1000.0*(t3-t0-(t2-t1)), 1000.0*(t2-t1)))

    return np.array(out_blocks)



def load_reference_noise(filename):
    if not os.path.exists(filename):
        return None
    fftsize = dct4stream.DEFAULT_DCT4_CFG.fft_size
    output = np.zeros((fftsize, 2))
    windows = []
    for block in dct4stream.AudioInputDCT4(filename):
        for window in block:
            if np.sum(np.abs(window)) != 0.0:
                windows.append(window)
    output = np.std(windows, axis=0)

    noise_floor = -51.50
    ref_value = np.std(scipy.fftpack.dct(np.ones((fftsize,)) * dB(noise_floor) * scipy.signal.windows.blackmanharris(fftsize), type=4, norm='ortho'))

    output = output / ref_value
    return output

def use(audio_filename, reference_noise_fn=None):

    model_path = "nrcnn.keras"

    if os.getenv("DNCNN_AUDIO_MODEL") is not None:
        model_path = os.getenv("DNCNN_AUDIO_MODEL")

    model: keras.Model = keras.models.load_model(model_path)
    reference_noise = load_reference_noise(reference_noise_fn)
    prev_block = None
    end = False
    input_path = pathlib.Path(audio_filename)
    output_path = input_path.with_stem(input_path.stem + ".denoised")


    with dct4stream2.AudioOutputDCT4(output_path) as output:
        for blocks in itertools.batched(dct4stream2.AudioInputDCT4(audio_filename), 8):
            inp_blocks = np.array(blocks) / reference_noise
            denoised_blocks = denoise_blocks(model, inp_blocks)
            denoised_blocks *= reference_noise
            for dblock in denoised_blocks:
                output.write_block(dblock)


def entry():
    dataset_dir = None
    audio_file = None
    dataset_source = None
    reference_noise = None
    opts, args = getopt.getopt(sys.argv[1:], "i:a:d:n:", ["idir=", "infile=", "dssrc=", "refnoise="])
    for opt, arg in opts:
        if opt in ("-i", "--idir"):
            dataset_dir = arg
        if opt in ("-a", "--infile"):
            audio_file = arg
        if opt in ("-d", "--dssrc"):
            dataset_source = arg
        if opt in ("-n", "--refnoise"):
            reference_noise = arg
    if dataset_dir is not None:
        if dataset_source is not None:
            preprocess_dataset(dataset_source, dataset_dir)
        else:
            train(dataset_dir)
    elif audio_file is not None:
        use(audio_file, reference_noise)


if __name__ == "__main__":
    # preprocess_dataset("/run/media/volt/d0p1_misc/dev/NoiseReductionNN/resources/TrainingSet/", "/run/media/volt/d1p0_misc2/denoise_dataset2/")
    # exit()
    entry()
