import time
from multiprocessing import Process
import subprocess

# Script to run and the number of cores
script_name = "multi_terminal_benchmarking.py"
num_cores = 7

# Start time for benchmarking
start_time = time.time()


# Function to run the script with an argument in a new terminal
def run_script(core_id):
    subprocess.Popen(["gnome-terminal", "--", "bash", "-c", f"python3 {script_name} {core_id}; exec bash"])


# Launch each script instance in parallel, each in a new terminal window
processes = []
for i in range(1, num_cores + 1):
    process = Process(target=run_script, args=(i,))
    processes.append(process)
    process.start()

# Wait for all processes to complete
for process in processes:
    process.join()

# End time and duration
end_time = time.time()
total_time = end_time - start_time
print(f"Total execution time: {total_time:.2f} seconds")
