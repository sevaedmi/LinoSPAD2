import os
import shutil
from glob import glob

from LinoSPAD2.functions import delta_t

path = r"D:\LinoSPAD2\Data\D2b_board\DCR\more data"
os.chdir(path)
files = glob("*.dat")

# Group the files by two
num_of_folders = len(files) // 2 + 1

new_folders = []

# Create new folders
for i in range(num_of_folders):
    new_folder = f"{2*i}-{2*i+1}"
    try:
        os.mkdir(new_folder)
    except FileExistsError as _:
        pass
    new_folders.append(f"{new_folder}")

# Copy the data files to the new folders
for i, file in enumerate(files):
    j = i // 2
    destination = os.path.join(path, new_folders[j], file)
    shutil.copyfile(file, destination)


###

# Collect the timestamp differences in eachf newly generated folder

for k, folder in enumerate(new_folders):
    # os.chdir(folder)

    delta_t.calculate_and_save_timestamp_differences(
        folder,
        pixels=[12, 58],
        rewrite=True,
        daughterboard_number="D2b",
        motherboard_number="#4",
        firmware_version="2212s",
        timestamps=1000,
        include_offset=False,
    )

    os.chdir("..")

###
