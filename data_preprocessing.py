import numpy as np
from pymongo import MongoClient
from scipy.io import loadmat
from tqdm import tqdm

client = MongoClient('mongodb://localhost:27017')
db = client.seizure_db

#query what seizures you want
structure = db.main.find({"FileLocation": {"$ne": None}, "FilePath": {"$ne": None}, "Structure": "Lateral posterior nucleus of the thalamus"}) #change value here

intialize_files = []
#count number of files to initialize numpy array
for seizure in structure:
    data_file = seizure['FileLocation']
    mat_file = loadmat(data_file)
    if 'BrainLocation' in mat_file and 'Recording' in mat_file:
        intialize_files.append(data_file)
    else:
        pass
num_files = len(intialize_files)
num_channels = 50 #change this value for a different amount of channels
sampling_rate = 1000
sequence_length = 60 * sampling_rate
data_np = np.full((num_files, num_channels, sequence_length), fill_value = 0)    
label_np = np.full((num_files, sequence_length), fill_value = 0)
file_idx = 0

#for loop to extract data and values
for seizure in enumerate(tqdm(structure, desc ="Seizure Progress")):
    data_file = seizure['FileLocation']
    mat_file = loadmat(data_file)
    if 'BrainLocation' in mat_file and 'Recording' in mat_file:
        channels = mat_file['BrainLocation']
        channel_data = mat_file['Recording']
        channel_data = np.array([row[0] for row in channel_data])
        channel_data = np.squeeze(channel_data)
        print(channel_data.shape)
        channel_indices = channels.flatten() == 'Lateral posterior nucleus of the thalamus'   #change value here
        selected_data = channel_data[channel_indices, :]
        selected_data = np.array(selected_data)
        print(selected_data.shape)
        #starting shaping data
        pre_seizure = seizure['PreSeizureDuration'] * sampling_rate 
        seizure_duration = seizure['SeizureDuration'] * sampling_rate
        post_seizure = seizure['PostSeizureDuration'] * sampling_rate

        full_seizure_duration = pre_seizure + seizure_duration + post_seizure
        max_pre_seizure = pre_seizure / 2   
        max_post_seizure = post_seizure / 2
        max_seizure_length = max_pre_seizure + seizure_duration + max_post_seizure
        #sequence shaping
        if max_seizure_length > sequence_length:
            amt_over = max_seizure_length - sequence_length
            cut_amt = amt_over/ 2
            min_pre_seiz = max_pre_seizure - (2 * sampling_rate)
            if cut_amt > min_pre_seiz:
                pre_cut_amt = min_pre_seiz
                post_cut_amt = amt_over - pre_cut_amt   
                start_time = (max_pre_seizure + pre_cut_amt)
                stop_time = (full_seizure_duration - max_post_seizure - post_cut_amt)
            elif cut_amt <= min_pre_seiz:
                pre_cut_amt = cut_amt
                post_cut_amt = cut_amt    
                
            start_time = (max_pre_seizure + pre_cut_amt)
            stop_time = (full_seizure_duration - max_post_seizure - post_cut_amt)
            selected_data = selected_data[:, start_time:stop_time]
        else:
            pass
             
            #channel shaping
        if selected_data[0] > num_channels:
            selected_data[0] = selected_data[:num_channels, :]
        else:
            pass
            
        #aggregate into final data vector
        data_np[file_idx, selected_data.shape[0], selected_data.shape[1]] = selected_data
        #label vector
        label_path = seizure['FilePath']
        label = np.load(label_path)
        label_np[file_idx] = label
        file_idx += 1
    else:
        print(f'Error: BrainLocation or Recording could not be found in file {data_file}')

print(data_np.shape)
print(label_np.shape)

np.save('/media/ben2X/SeizureBot/data_vector.npy', data_np)
np.save('/media/ben2X/SeizureBot/label_vector.npy', label_np)


       




            
    







