import numpy as np
from pymongo import MongoClient

client = MongoClient()
database = client.seizure_db

file_list = list(database.main.find({"FilePath" : {"$ne", None}}))

window_size = 2000
label_vector = []

for file in file_list:
    vector = np.load(file)
    window_start_idx = None
    for i in range(0, len(vector)-window_size, 1):
       window = vector[i:i + window_size]
       if np.all(window == 1):
          window_start_idx = i
          break
    if window_start_idx == None:
        continue
    
    real_window = [1]
    forward_window = []
    backward_window = []

    forward_idx = window_start_idx + window_size
    while forward_idx <= len(vector):
       forward_window.append(0)
       forward_idx += window_size

    backward_idx = window_start_idx - window_size
    while backward_idx >= 0:
       backward_window.append(0)
       backward_idx -= 2000

    label_vector = backward_window + real_window + forward_window
    np.save(file, np.array(label_vector))


       