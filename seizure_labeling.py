from pymongo import MongoClient
import numpy as np

client = MongoClient('mongodb://localhost:27017')
db = client.seizure_db

valid_files = db.main.find({"FileLocation": {"$ne": None}})



#for loop
for seizure in valid_files:
    total_timepoints = int((seizure['DataCollectionEnd'] - seizure['DataCollectionStart']) * 1000) 
    label_vector = np.zeros(total_timepoints, dtype=int)
    seizure_start_time = seizure['SeizureStartTime']
    seizure_start_index = int((seizure_start_time - seizure['DataCollectionStart']) * 1000)
    start_index = max(0, seizure_start_index - 2000) 
    label_vector[start_index:seizure_start_index] = 1
    file_path = f"/media/ben2X/SeizureBot/labeled_seizures/{seizure['_id']}.npy"
    np.save(file_path, label_vector)
    db.main.update_one(
        {"_id": seizure["_id"]},
        {"$set": {"FilePath": file_path}})
