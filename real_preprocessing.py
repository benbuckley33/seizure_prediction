import numpy as np
from pymongo import MongoClient
from scipy.io import loadmat


client = MongoClient('mongodb://localhost:27017')
db = client.seizure_db

#query what seizures you want
structure = list(db.main.find({"FileLocation": {"$ne": None}, "FilePath": {"$ne": None}, "Structure": "Lateral posterior nucleus of the thalamus"})) #change value here


intialize_files = []
data_vector = []
label_vector = []
backward_vector = []
forward_vector = []
min_channels = 5

#count number of files to initialize numpy array
for idx, seizure in enumerate(structure, 1):
    print(f"Processing seizure number: {idx}")
    data_file = seizure['FileLocation']
    mat_file = loadmat(data_file)
    if 'BrainLocation' in mat_file and 'Recording' in mat_file:
        intialize_files.append(data_file)
        channels = mat_file['BrainLocation']
        channel_data = mat_file['Recording']
        channel_data = np.array([row[0] for row in channel_data])
        channel_data = np.squeeze(channel_data)
        print(channel_data.shape)
        channel_indices = channels.flatten() == 'Lateral posterior nucleus of the thalamus'   #change value here
        selected_data = channel_data[channel_indices, :]
        
        data_collection_start = seizure['DataCollectionStart']
        seizure_start = seizure['SeizureStartTime'] 
        label_path = seizure['FilePath']

        norm_start = seizure_start - data_collection_start
        preictal_start = norm_start - 2
        split_start = round(preictal_start * 1000)
        preictal_window = selected_data[:5, split_start: (split_start + 2000)]
        if preictal_window.shape != (5, 2000):
            print(f'Skipping due to bad preictal window shape {preictal_window.shape}')
        predicted_idx = int(preictal_start * 1000 // 2000)
        label_nump = np.load(label_path)
        real_window = np.where(label_nump == 1)[0]
        print(f'selected DATA shape: {selected_data.shape}')
        if selected_data.shape[0] >= 5 and predicted_idx in real_window:
            selected_data = selected_data[:5, :]
        
            backward_idx = split_start - 2000
            while backward_idx >= 0:
                segment = selected_data[:, backward_idx:backward_idx + 2000]
                if segment.shape == (5, 2000):
                    backward_vector.append(segment)
                backward_idx -= 2000
            forward_idx = split_start + 2000
            while forward_idx <= selected_data.shape[1]:
                segment = selected_data[:, forward_idx:forward_idx + 2000]
                if segment.shape == (5, 2000):
                    forward_vector.append(segment)
                forward_idx += 2000
            for i in reversed(range(len(backward_vector))):
                data_vector.append(backward_vector[i])
            data_vector.append(preictal_window)
            for i in forward_vector:
                data_vector.append(i)
            label_vector.append(label_nump)
            backward_vector.clear()
            forward_vector.clear()
            print(f'{data_vector[-1].shape}') 
        else:
            print(f'Error: File {data_file} has fewer than minimum channels necessary or doesnt match label')
            continue
    else:
        pass
    print(f'File {len(intialize_files)}')   
else:
    pass
print(f"Shape of data_vector: ({len(data_vector)}, {data_vector[0].shape[0]}, {data_vector[0].shape[1]})")

# Sanity check for all data_vector elements
bad_shapes = []
for i, item in enumerate(data_vector):
    if item.shape != (5, 2000):
        print(f"Bad shape at index {i}: {item.shape}")
        bad_shapes.append((i, item.shape))

print(f"Total mismatched entries: {len(bad_shapes)}")

np.save('/media/ben2X/SeizureBot/data_vector.npy', data_vector)
np.save('/media/ben2X/SeizureBot/label_vector.npy', label_vector)

load_data = np.load('/media/ben2X/SeizureBot/data_vector.npy')
load_label = np.load('/media/ben2X/SeizureBot/label_vector.npy')

print(load_data.shape)
print(load_label.shape)
