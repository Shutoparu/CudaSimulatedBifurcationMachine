import numpy as np
import numpy.ctypeslib as ctplib
from ctypes import c_float, c_int, cdll, POINTER
import time

np.random.seed(1)

dim = 10
window = 5
maxStep = 600
qubo = 2 * np.random.rand(dim, dim).astype(np.float32) - 1
qubo = (qubo + qubo.T) / 2
np.savetxt("qubo.txt", qubo)
# quit()
qubo = qubo.flatten()
spin = 2 * np.random.rand(dim, dim).astype(np.float32) - 1

# # test code
# dim = 5
# window = 50
# maxStep = 10000
# qubo = np.array([[-.1, .2, -.3, .4, -.5], [.2, .3, -.4, .5, .6], [-.3, -.4, -.5, -.6, -.7],
#                 [.4, .5, -.6, .7, .8], [-.5, .6, -.7, .8, -.9]]).astype(np.float32)
# qubo = qubo.flatten()
# spin = 2 * np.random.rand(dim).astype(np.float32) - 1
# # test code
for i in range(dim):
    spin[i] = ctplib.as_ctypes(spin[i])
qubo = ctplib.as_ctypes(qubo)

sbm = cdll.LoadLibrary("./lib/sbm.so")

main = sbm.iterate

main.argtypes = [POINTER(c_float), POINTER(c_float), c_int, c_int, c_int]
# main.restype = c_float

start = time.time()
# energy = main(spin, qubo, dim, window, maxStep)
for i in range(dim):
    # main(spin[i], qubo, dim, window, maxStep)
    main(ctplib.as_ctypes(spin[i]), qubo, dim, window, maxStep)
end = time.time()

for i in range(dim):
    spin[i] = ctplib.as_array(spin[i])
    spin[i] = np.sign(spin[i])
    
qubo = np.reshape(qubo, [dim, dim])

energy = list()
for i in range(dim):
    energy.append(-.5 * (np.expand_dims(spin[i],axis=1).T @ qubo @ np.expand_dims(spin[i],axis=1))[0][0])

print(min(energy))
# print(spin)
print("spent time: ", end-start)

# # test code
# binary = np.expand_dims(binary, axis=1)
# print( - binary.T @ qubo @ binary)
# # test code
