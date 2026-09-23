import numpy as np
import matplotlib.pyplot as plt


raw = np.fromfile("data/d18011.001.dat", dtype=np.int16)
data = raw.reshape(-1,8)
#2 function for looping into the .data file and extracting the voltage values for each channel  
...
#3 set up a threshold for spike detection and implement a spike detection algorithm (e.g., thresholding, template matching, or machine learning-based methods) to identify the timestamps of spikes in the voltage data.
#pull certain channels inside each intracelluar and extracelluar
intracellular = data[:, 6]   # membrane potential — your ground truth
extracellular = data[:, 1]   # pick one of channels 0–5 to start with


print(len(extracellular))