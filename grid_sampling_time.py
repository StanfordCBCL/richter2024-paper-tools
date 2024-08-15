import numpy as np

times = []
with open("grid_sampling_time.txt") as ff:
    for line in ff.readlines():
        time = float(line.split(" ")[3])
        times.append(time)

print(len(times))
print(np.sum(times) / 3600)