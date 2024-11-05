import multiprocessing
import time


# CPU-intensive task: Sum of squares calculation
def sum_of_squares(n):
    return sum(i * i for i in range(n))


# Function to run with a single core
def single_core_computation(numbers):
    results = []
    for number in numbers:
        results.append(sum_of_squares(number))

    # write results to a file
    with open("tmp_result.txt", "a") as file:
        for result in results:
            file.write(f"{result}\n")
    return results


# Function to run with multiprocessing pool
def multi_core_computation(numbers):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(sum_of_squares, numbers)
    return results


if __name__ == "__main__":
    # Range of large numbers for heavy computation
    numbers = [10_000_000 + i for i in range(20)]

    # Single-core computation
    start_time = time.time()
    single_core_results = single_core_computation(numbers)
    single_core_time = time.time() - start_time
    print(f"Single-core computation time: {single_core_time:.2f} seconds")

    # Multi-core computation using Pool
    start_time = time.time()
    multi_core_results = multi_core_computation(numbers)
    multi_core_time = time.time() - start_time
    print(f"Multi-core computation time: {multi_core_time:.2f} seconds")

    # Display speedup
    speedup = single_core_time / multi_core_time
    print(f"Speedup: {speedup:.2f}x")
