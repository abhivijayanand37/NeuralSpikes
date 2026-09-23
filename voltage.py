import numpy as np

def recovery_period(s, upper=55, lower=45):
    #goes through each element of array
    current = s[:-1]
    next_val = s[1:]

    #comparing values to minimum voltages needed to cause a spike
    #looks at relative and absolute refractory period to see if 1 or 2 spikes fire
    relative_mask = (current <= upper) & (next_val <= lower)
    absolute_mask = (current <= upper) & (next_val > lower)

    relative_idx = np.where(relative_mask)[0]
    absolute_idx = np.where(absolute_mask)[0]

    #prints where relative and absolute index appears
    for i in relative_idx:
        print(f"Relative Refractory - 2 Spikes at index {i}, voltages = {s[i]}, {s[i+1]}")

    for i in absolute_idx:
        print(f"Absolute Refractory - 1 Spike at index {i}, voltage = {s[i]}")

    return relative_idx, absolute_idx


voltages = np.random.randint(26, 101, size=30)
print("Voltages:", voltages)

recovery_period(voltages)