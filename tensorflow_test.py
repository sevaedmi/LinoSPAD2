import tensorflow as tf
import time
from pathlib import Path
import glob
import os
import numpy as np


def benchmarking():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled.")
        except RuntimeError as e:
            print("Error enabling GPU memory growth:", e)

    # Get the current script directory
    current_directory = Path(__file__).parent
    # Define the path to the 'raw_data' directory
    path = str(current_directory / 'tmp_raw_data')

    # Find all .dat files in the directory
    files = glob.glob(os.path.join(path, "*.dat*"))

    # read every file in the directory using np.memmap
    tmp_results = []
    start = time.time()
    for file in files:
        raw_data = np.memmap(file, dtype=np.uint32)
        tmp_results.append(raw_data)
    stop = time.time()
    print(f"Reading {len(tmp_results)} files using np.memmap took {round(stop - start, 2)} s")

    # read every file in the directory using np.fromfile
    tmp_results = []
    start = time.time()
    for file in files:
        raw_data = np.fromfile(file, dtype=np.uint32)
        tmp_results.append(raw_data)
    stop = time.time()
    print(f"Reading {len(tmp_results)} files using np.fromfile took {round(stop - start, 2)} s")

    # read every file in the directory using tensorflow batched reading
    start = time.time()
    file_dataset = tf.data.Dataset.from_tensor_slices(files)
    # Read and process each file in the dataset
    raw_data_dataset = file_dataset.map(
        lambda file: tf.io.read_file(file),
        num_parallel_calls=tf.data.AUTOTUNE  # Use parallelism for efficiency
    )
    stop = time.time()
    print(f"Reading files using tensorflow batched reading took {round(stop - start, 2)} s")


if __name__ == "__main__":
    benchmarking()


