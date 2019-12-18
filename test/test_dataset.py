import scipy.io as sio

from utils.dataset import *

d = Dataset()
data, label = d.generate_gmm()
data = d.generate_circle()
sio.savemat('data.mat', {'data': data})
print(d.next_batch())
print(data, label)
