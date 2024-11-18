import tensorflow as tf
import glob
import os
import numpy as np
import time



# Define a function to load and process a single file
def process_file(file_path):
    raw_data = tf.io.read_file(file_path)
    return raw_data


# Create a TensorFlow dataset that reads files in batches
def data_generator(files, batch_size=32):
    # Create a TensorFlow dataset from the list of file paths
    file_dataset = tf.data.Dataset.from_tensor_slices(files)

    # Map each file to a processing function (using process_file)
    raw_data_dataset = file_dataset.map(
        lambda file: process_file(file),
        num_parallel_calls=tf.data.AUTOTUNE  # Parallelize file reading
    )

    # Batch the dataset for batching the data during processing
    batched_dataset = raw_data_dataset.batch(batch_size)

    # Optional: Prefetch for better performance
    batched_dataset = batched_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return batched_dataset


# Example usage
def benchmarking():
    # Define the path to your data
    path = 'tmp_raw_data'

    # Find all .dat files in the directory
    files = glob.glob(os.path.join(path, "*.dat"))

    # Create the data generator (batches of 32 files)
    batch_size = 32
    batched_data = data_generator(files, batch_size)

    # Iterate through batches
    start = time.time()
    for batch in batched_data:
        # You can process each batch here
        print(f"Processing batch with {len(batch)} items")

    stop = time.time()
    print(f"Processing batches took {round(stop - start, 2)} s")


if __name__ == "__main__":
    benchmarking()
