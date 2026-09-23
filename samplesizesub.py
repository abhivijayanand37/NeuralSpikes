
import numpy as np
import matplotlib.pyplot as plt
#input what sample size to analyze and what channel to analyze for extracellular data.)
channel = int(input("Enter the channel number (0-5) for extracellular data: "))

#Obtain .data file for analysis from hc1 and .xml file for metadata from hc1
raw = np.fromfile("data/d18011.001.dat", dtype=np.int16)
data = raw.reshape(-1,8)

#Pull certain channels based on the metadata in the .xml file for analysis
intracellular = data[:, 6]   # membrane potential 
extracellular = data[:, channel]   # pick one of channels 0–5 to start with

#base noise level for channel.
mad_e = np.median(np.abs(extracellular - np.median(extracellular))) / 0.6745
mad_i = np.median(np.abs(intracellular - np.median(intracellular))) / 0.6745

#Set Threshold for spike detection and time data 
threshold_e = np.median(extracellular) - 5*mad_e
threshold_i = np.median(intracellular) + 10*mad_i

detected_spike_indices_intracellular = []

# Where the results of #spikes are returned as well as the refractory period used.
spikes_results = []
refractory_period = []

# Loop different threholds to find points crossing the threshold (ignoring consecutive points from the same spike)

#Precision and Accuracy base
precision = []
accuracy = []

i = 0
while i < len(intracellular):
    if intracellular[i] > threshold_i:
        detected_spike_indices_intracellular.append(i)
        i += 80  # Skip the next 4ms at 20000Hz(refractory period) to avoid double-counting the same spike
    else:
        i += 1

for refractory in range(20, 500, 20):  # Loop 20-500 in 20ms increments to find the optimal refractory period
    detected_spike_indices_extracellular = []
    i = 0
    while i < len(extracellular):
        if extracellular[i] < threshold_e:
            detected_spike_indices_extracellular.append(i)
            i += refractory
        else:
            i += 1
    refractory_period.append(refractory)
    spikes_results.append(len(detected_spike_indices_extracellular))
    
    #Precision and Accuracy calculations.
    distance = np.array(detected_spike_indices_extracellular)[:, None] - np.array(detected_spike_indices_intracellular)[None, :]
    true_positives = np.sum(np.any(np.abs(distance) <= 20, axis=1))  # Count true positives within 1ms window
    #finds true and false postiives/negatives.
    false_positives = len(detected_spike_indices_extracellular) - true_positives
    false_negatives = len(detected_spike_indices_intracellular) - (np.sum(np.any(np.abs(distance) <= 20, axis=0)))  # Count false negatives within 1ms window
    #appends to precions and accuracy lists for graphing
    precision.append(true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0)
    accuracy.append(true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0)


#Convert to arrays for time and general conversions 
detected_spike_indices_extracellular = np.array(detected_spike_indices_extracellular)
detected_spike_indices_intracellular = np.array(detected_spike_indices_intracellular)


# Time axis (same length as both sliced channels now)
time = np.linspace(0, len(extracellular) / 20000, len(extracellular))
print("-------Resutlts--------")
print("Extracellular spikes detected:", len(detected_spike_indices_extracellular))
print("Intracellular spikes detected:", len(detected_spike_indices_intracellular))
print("Thresholds tested:", threshold_e)
print("Accuracy results for each Threshold:", spikes_results)
print("Precision for each Threshold:", precision)
print("-----Metrics-----")

# --- Plotting ---
#Plots # of spikes detected vs refractory period for extracellular data
plt.savefig("channel_" + str(channel) + "_threshold_detection_metrics.png")

print("-------Resutlts--------")
print("Extracellular spikes detected:", len(detected_spike_indices_extracellular))
print("Intracellular spikes detected:", len(detected_spike_indices_intracellular))
print("Refractory periods tested:", refractory_period)
print("Accuracy results for each Threshold:", spikes_results)
print("Precision for each Threshold:", precision)
print("-----Metrics-----")

# --- Plotting ---
#Plots # of spikes detected vs refractory period for extracellular data
 
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharex=True)

#ax1 shows the graph of spikes detected with refractory period.
ax1.plot(refractory_period, spikes_results, marker='o', color='blue')
ax1.set_title("Number of Spikes vs. Refractory Period")
ax1.set_xlabel("Refractory Period (samples)")
ax1.set_ylabel("Number of Spikes Detected")

ax2.plot(refractory_period, precision, marker='o', color='green')
ax2.set_title("Precision vs. Refractory Period")
ax2.set_xlabel("Refractory Period (samples)")
ax2.set_ylabel("Precision")

ax3.plot(refractory_period, accuracy, marker='o', color='red')
ax3.set_title("Accuracy vs. Refractory Period")
ax3.set_xlabel("Refractory Period (samples)")
ax3.set_ylabel("Accuracy")

plt.savefig("channel_" + str(channel) + "_spike_detection_metrics.png")

