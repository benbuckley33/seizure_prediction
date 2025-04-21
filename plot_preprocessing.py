import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

# Dummy load (replace with your real data)
data = np.load('/media/ben2X/SeizureBot/data_vector.npy')        
labels = np.load('/media/ben2X/SeizureBot/label_vector.npy')    

i = 687
data_plt = data[i,0,:]
timepoints = list(range(0,2000))
segment = labels[i]
label_idx = np.where(labels == 1)[0]


plt.plot(timepoints, data_plt)
plt.xlabel("Timepoints")
plt.ylabel("Voltage")

plt.savefig("/media/ben2X/SeizureBot/seg_plot.png")
print("plot saved")
