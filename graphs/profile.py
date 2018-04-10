import numpy as np
import sys
from matplotlib import pyplot as plt


def read_datafile(file_name):
    # the skiprows keyword is for heading, but I don't know if trailing lines
    # can be specified
    data = np.loadtxt(file_name, delimiter=',', skiprows=1)
    return data

filename = sys.argv[1]
print(filename)
data = read_datafile(filename)
x = data[:,0]
y = data[:,4]
plt.plot(x,y)
plt.title('Karp Flatt Measue vs Number of Processors')
plt.ylabel('Karp Flatt Measue')
plt.xlabel('Number of Processors')

# plt.show()
imagefile = "karpflatt_"+filename.split(".")[0]+".png"
plt.savefig(imagefile)
