import numpy as np
import autograd.numpy as np
from autograd import grad
# import matplotlib.pyplot as plt

def binaryEntropy (p):
    # return p
    return -p*np.log2(p) -(1-p)*np.log2(1-p)

grad_ent = grad(binaryEntropy)
print "Gradient of binaryEntropy(p) is", grad_ent(0.5)


# pp = np.linspace(0,1,50)
# plt.plot(pp, ent(pp))
# plt.show()







