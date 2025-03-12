from scipy.io import loadmat
from pymongo import MongoClient
import numpy as np

data = loadmat('/media/ben2X/database/raw_channel_data/0023/2024_01_23__23_15_31/channelData_mouse0023__SWD0001_converted.mat')
client = MongoClient('mongodb://localhost:27017')
db = client.seizure_db

#this needs to be changed to go through every seizure we have
seizure_info = db.main.find_one({"SeizureNumber": 1})

total_timepoints = int((seizure_info['DataCollectionEnd'] - seizure_info['DataCollectionStart']) * 1000) 
label_vector = np.zeros(total_timepoints, dtype=int)
seizure_start_time = seizure_info['SeizureStartTime']
seizure_start_index = int((seizure_start_time - seizure_info['DataCollectionStart']) * 1000)
start_index = max(0, seizure_start_index - 2000) 
label_vector[start_index:seizure_start_index] = 1

print(f"part of vector: {label_vector[start_index-2:seizure_start_index+2]}")
print(f"total timepoints: {total_timepoints}")
print(f"start index (2 seconds/2000 points before seizure start): {start_index}")
print(f"end index (seizure start): {seizure_start_index}")
print(f"amount of 1's (should be 2000): {np.sum(label_vector)}")