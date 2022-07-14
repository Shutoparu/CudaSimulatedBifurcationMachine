import numpy as np
import numpy.ctypeslib as ctplib
from ctypes import c_float, c_int, cdll, POINTER
import time



# dim = 1050
# window = 50
# maxStep = 100000
# qubo = 2 * np.random.rand(dim, dim).astype(np.float32) - 1
# qubo = (qubo + qubo.T) / 2
# qubo = qubo.flatten()
# spin = 2 * np.random.rand(dim).astype(np.float32) - 1

# test code
dim = 5
window = 50
maxStep = 10000
qubo = np.array([[-.1, .2, -.3, .4, -.5], [.2, .3, -.4, .5, .6], [-.3, -.4, -.5, -.6, -.7],
                [.4, .5, -.6, .7, .8], [-.5, .6, -.7, .8, -.9]]).astype(np.float32)
qubo = qubo.flatten()
spin = 2 * np.random.rand(dim).astype(np.float32) - 1
# test code

spin = ctplib.as_ctypes(spin)
qubo = ctplib.as_ctypes(qubo)

sbm = cdll.LoadLibrary("./lib/sbm.so")

main = sbm.iterate

main.argtypes = [POINTER(c_float), POINTER(c_float), c_int, c_int, c_int]
# main.restype = c_float

start = time.time()
# energy = main(spin, qubo, dim, window, maxStep)
main(spin, qubo, dim, window, maxStep)
end = time.time()

spin = ctplib.as_array(spin)
spin = np.expand_dims(np.sign(spin), axis=1)
qubo = np.reshape(qubo, [dim, dim])

energy = -spin.T @ qubo @ spin

print(energy)
print(spin)
print("spent time: ", end-start)

# # test code
# binary = np.expand_dims(binary, axis=1)
# print( - binary.T @ qubo @ binary)
# # test code
