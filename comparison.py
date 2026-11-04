import matplotlib.pyplot as plt
import numpy as np

channels_x_axis = [0, 1, 2, 3, 4, 5]  

d180111_f1_values = [0.1296, 0.0183, 0.0000, 0.0124, 0.0493, 0.1442]  
d180112_f1_values = [0.1708, 0.0374, 0.0051, 0.0047, 0.0692, 0.1865]  
d180113_f1_values = [0.0860, 0.0174, 0.0091, 0.0043, 0.0359, 0.0887]  
d180114_f1_values = [0.1217, 0.0119, 0.0011, 0.0035, 0.0442, 0.1433]  
d180115_f1_values = [0.1869, 0.0345, 0.0054, 0.0114, 0.0871, 0.1826]  
  # Read from Physical Chart 1 (e.g., Left Axis scale)


# --- STEP 2: Combine them onto one dual-axis canvas ---
fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot 1 data
color1 = '#1f77b4' # Blue
plt.yticks(np.arange(0, 0.2, 0.02))
ax1.set_xlabel('Channels', fontweight='bold')
ax1.set_ylabel('F1 Score', color='black', fontweight='bold')
line1 = ax1.plot(channels_x_axis, d180111_f1_values, color=color1, marker='x', linewidth=2, label='Chart 1')
ax1.tick_params(axis='y', labelcolor=color1)

# Plot 2 data
color2 = 'red'
line1 = ax1.plot(channels_x_axis, d180112_f1_values, color=color2, marker='o', linewidth=2, label='Chart 1')
ax1.tick_params(axis='y', labelcolor=color2)

# Plot 3 data
color3 = 'black'
line1 = ax1.plot(channels_x_axis, d180113_f1_values, color=color3, marker='*', linewidth=2, label='Chart 1')
ax1.tick_params(axis='y', labelcolor=color2)


# Plot 4 data
color4 = 'green'
line1 = ax1.plot(channels_x_axis, d180114_f1_values, color=color4, marker='*', linewidth=2, label='Chart 1')
ax1.tick_params(axis='y', labelcolor=color3)

# Plot 5 data
color5 = 'green'
line1 = ax1.plot(channels_x_axis, d180114_f1_values, color=color5, marker='*', linewidth=2, label='Chart 1')
ax1.tick_params(axis='y')




plt.title('f1 on Filtered Data', fontsize=14, fontweight='bold', pad=15)
plt.savefig("ComparisonTest.png")
