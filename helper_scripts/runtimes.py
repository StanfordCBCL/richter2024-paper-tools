import numpy as np

runtimes = []

with open("runtimes.txt") as ff:

    for line in ff.readlines():
        splitline = line.split(" ")
        for i, field in enumerate(splitline):
            if field == "in":
                runtime = float(splitline[i+1])
                runtimes.append(runtime)

print(np.sum(runtimes))
print(np.mean(runtimes))