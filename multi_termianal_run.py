import subprocess
import time

# Script to run and the number of cores
script_name = "multi_terminal_benchmarking.py"
num_cores = 10

# Start time for benchmarking
start_time = time.time()

# Launch each script instance in a new command prompt
processes = []
for i in range(1, num_cores + 1):
    # 'cmd' opens a new command prompt window for each instance
    cmd = ["cmd", "/c", "start", "python", script_name, str(i)]
    process = subprocess.Popen(cmd)
    processes.append(process)

# Wait for all processes to complete
for process in processes:
    process.wait()

# End time and duration
end_time = time.time()
total_time = end_time - start_time
print(f"Total execution time: {total_time:.2f} seconds")
