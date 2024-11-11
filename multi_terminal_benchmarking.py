import os
import time
import sys
from src.LinoSPAD2.functions import delta_t
from pathlib import Path

serial_number_of_script = sys.argv[1]  # "chunk" counter for the script

current_directory = str(Path(__file__).parent)
path = str(current_directory + '/raw_data/script_' + str(serial_number_of_script))
start = time.time()
delta_t.calculate_and_save_timestamp_differences_fast(
    path,
    pixels=[144, 171],
    rewrite=True,
    daughterboard_number="NL11",
    motherboard_number="#33",
    firmware_version="2212b",
    timestamps=300,
    include_offset=False,
)
finish = time.time()
print(f"{finish - start} s")

# Find all .feather files in the directory
path = str(path + '/delta_ts_data')
feather_files = [path + '/' + f for f in os.listdir(path) if f.endswith('.feather')]

# Delete all the feather files
for file in feather_files:
    os.remove(file)

print("Deleted all the feather files")
