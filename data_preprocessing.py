import numpy as np
from pymongo import MongoClient
from scipy.io import loadmat

client = MongoClient('mongodb://localhost:27017')
db = client.seizure_db

#query what seizures you want
structure = db.main.find({"FileLocation": {"$ne": None}, "FilePath": {"$ne": None}, "Structure": "thalamus"}) #change value here

data_list = []
label_list = []

#for loop to extract data and values
for seizure in structure:
    data_file = seizure['FileLocation']
    mat_file = loadmat(data_file)
    if 'BrainLocation' in mat_file and 'Recording' in mat_file:
        channels = mat_file['BrainLocation']
        channel_data = mat_file['Recording']
        channel_indices = channels == 'thalamus'   #change value here
        selected_data = channel_data[channel_indices]
        selected_nump = np.array(selected_data)
        data_list.append(selected_nump)
        label_path = seizure['FilePath']
        label = np.load(label_path)
        label_list.append(label)
    else:
        print(f'Error: BrainLocation or Recording could not be found in file {data_file}')

data_vector = np.stack(data_list, axis = 0)
label_vector = np.stack(label_list, axis = 0)

print(label_vector.shape)
print(data_vector.shape)

np.save('/media/ben2X/SeizureBot/data_vector.npy', data_vector)
np.save('/media/ben2X/SeizureBot/label_vector.npy', label_vector)








