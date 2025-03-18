import numpy as np
import pandas as pd
import scipy.io as sio

# Load the .mat file
file_path = "/media/ben2X/database/raw_channel_data/0024/2024_01_23__23_35_34/channelData_mouse0024__SWD0001_converted.mat" 
mat_data = sio.loadmat(file_path)
print("Kets in mat_data:", mat_data.keys())
brain_location = mat_data['BrainLocation']
recording = mat_data['Recording']

# Assuming brain_location is an array and we are filtering by a specific brain location
target_location = 'Field CA1'

# Find indices where BrainLocation matches the target_location
indices = [i for i, location in enumerate(brain_location) if location == target_location]

# Print corresponding Recording values for those indices
for idx in indices:
    print(recording[idx])
