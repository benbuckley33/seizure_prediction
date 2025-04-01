import numpy as np
from pymongo import MongoClient
from scipy.io import loadmat

client = MongoClient('mongodb://localhost:27017')
db = client.seizure_db

#query what seizures you want
structure = db.main.find({"FileLocation": {"$ne": None}, "FilePath": {"$ne": None}, "Structure": "Lateral posterior nucleus of the thalamus"}) #change value here

data_list = []
label_list = []

#for loop to extract data and values
for seizure in structure:
    data_file = seizure['FileLocation']
    mat_file = loadmat(data_file)
    if 'BrainLocation' in mat_file and 'Recording' in mat_file:
        channels = mat_file['BrainLocation']
        channel_data = mat_file['Recording']
        channel_indices = channels == 'Lateral posterior nucleus of the thalamus'   #change value here
        selected_data = channel_data[channel_indices]
        selected_nump = np.array(selected_data) #from here on should be changed by the timepoint and channel shaping code
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

#making timepoint and channel dimension homogenous
num_channels = 50 #change this value for a different amount of channels
full_seizure_duration = seizure['PreSeizureDuration'] + seizure['SeizureDuration'] + seizure['PostSeizureDuration']
max_pre_seizure = (seizure['PreSeizureDuration'])/2
max_post_seizure = (seizure['PostSeizureDuration'])/2
seizure_duration = seizure['SeizureDuration']
max_seizure_length = max_pre_seizure + seizure_duration + max_post_seizure
    
for channels in selected_data:
    if channels >= num_channels:
        #only take 50 channels
        if max_seizure_length > 60:
            amt_over = max_seizure_length - 60
            cut_amt = amt_over/2
            min_pre_seiz = max_pre_seizure - 2
            if cut_amt > min_pre_seiz:
                pre_cut_amt = min_pre_seiz
                post_cut_amt = amt_over - pre_cut_amt   #i think here im not controlling for post time being too short
                start_time = (max_pre_seizure + pre_cut_amt) * 1000
                top_time = (full_seizure_duration - max_post_seizure - post_cut_amt) * 1000
            elif cut_amt <= min_pre_seiz:
                start_time = (max_pre_seizure + cut_amt) * 1000
                stop_time = (full_seizure_duration - max_post_seizure - cut_amt) * 1000
        elif max_seizure_length <= 60:
        #fill the rest of the vector with 0s until it reaches 60 seconds
    elif channels < num_channels:
        #create padded channels until there are 50 of length 60000

            
    







