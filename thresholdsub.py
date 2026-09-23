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

threshold_i = np.median(intracellular) + 10 * (np.median(np.abs(intracellular - np.median(intracellular))) / 0.6745)

detected_spike_indices_extracellular = []
detected_spike_indices_intracellular = []

#Precision and Accuracy base
spikes_results = []
precision = []
thresholds = []

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
# Loop to find points crossing the threshold (ignoring consecutive points from the same spike)

window_size = 20000  # 1 second window at 20kHz
for window_start in range(0, 1500000, 20000):  # Loop through the data in 1-second windows (20,000 samples at 20kHz)
    window_data = extracellular[window_start : window_start + window_size]

    median_e = np.median(window_data)
    mad_e = np.median(np.abs(window_data - median_e)) / 0.6745
    threshold_e = median_e - 5 * mad_e
    threshold_e = np.median(extracellular) - 5*mad_e

    detected_spike_indices_extracellular = []
    i = 0
    while i < len(extracellular):
        if extracellular[i] < threshold_e:
            detected_spike_indices_extracellular.append(i)
            i += 80
        else:
            i += 1
    thresholds.append(threshold_e)
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

# Convert indices back to timestamps in seconds
#Convert to arrays for time and general conversions 
detected_spike_indices_extracellular = np.array(detected_spike_indices_extracellular)
detected_spike_indices_intracellular = np.array(detected_spike_indices_intracellular)

#Addes back wind
detected_spike_indices_extracellular = [idx + window_start for idx in detected_spike_indices_extracellular]

# Time axis (same length as both sliced channels now)
time = np.linspace(0, len(extracellular) / 20000, len(extracellular))

print("Extracellular spikes detected:", len(detected_spike_indices_extracellular))
print("Intracellular spikes detected:", len(detected_spike_indices_intracellular))

# --- Plotting ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharex=True)

# Time axis (same length as both sliced channels now)
time = np.linspace(0, len(extracellular) / 20000, len(extracellular))
print("-------Resutlts--------")
print("Extracellular spikes detected:", len(detected_spike_indices_extracellular))
print("Intracellular spikes detected:", len(detected_spike_indices_intracellular))
print("Thresholds tested:", threshold_e)
print("Spikes results for each Threshold:", spikes_results)
print("Precision for each Threshold:", precision)
print("-----Metrics-----")

# --- Plotting ---
#Plots # of spikes detected vs refractory period for extracellular data
 
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharex=True)

#ax1 shows the graph of spikes detected with refractory period.
ax1.plot(thresholds, spikes_results, marker='o', color='blue')
ax1.set_title("Number of Spikes vs. Subsets")
ax1.set_xlabel("ADC Values")
ax1.set_ylabel("Number of Spikes Detected")

ax2.plot(thresholds, precision, marker='o', color='green')
ax2.set_title("Precision vs. Subsets")
ax2.set_xlabel("ADC Values")
ax2.set_ylabel("Precision")

ax3.plot(thresholds, accuracy, marker='o', color='red')
ax3.set_title("Accuracy vs. Subsets")
ax3.set_xlabel("ADC Values")
ax3.set_ylabel("Accuracy")

plt.tight_layout()
plt.savefig("channel_" + str(channel) + "_subset_detection_metrics.png")




