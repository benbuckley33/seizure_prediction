from scipy.io import loadmat

file_path = '/media/ben2X/database/raw_channel_data/0027/2024_01_24__00_37_32/channelData_mouse0027__SWD0002_converted.mat'


final_vector = []
mat_file = loadmat(file_path)
print(mat_file['Recording'])
print(type(mat_file['Recording']))  # See if it's an array or object
print(mat_file['Recording'].dtype)  # Check the data type
print(mat_file['Recording'].shape)  # Confirm its shape
channels = mat_file['BrainLocation']
channel_data = mat_file['Recording']
channel_indices = channels == 'fiber tracts'   #change value here
selected_data = channel_indices == channel_data
final_vector.append(selected_data)
print(final_vector.shape)