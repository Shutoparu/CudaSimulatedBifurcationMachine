import numpy as np
import numpy.ctypeslib as ctplib
from ctypes import c_float, c_int, cdll, POINTER
import time

np.random.seed(1)
dim = 750
window = 50
maxStep = 100000
qubo = 2 * np.random.rand(dim, dim).astype(np.float32) - 1
qubo = (qubo + qubo.T) / 2
# quit()
qubo = qubo.flatten()
np.random.seed()
spin = 2 * np.random.rand(dim).astype(np.float32) - 1

# # test code
# dim = 2
# window = 10
# maxStep = 1000
# qubo = np.array([[0,1],[1,0]]).astype(np.float32)
# qubo = qubo.flatten()
# spin = 2 * np.random.rand(dim).astype(np.float32) - 1
# spin = np.array([.0,.0]).astype(np.float32)
# # test code

spin = ctplib.as_ctypes(spin)
qubo = ctplib.as_ctypes(qubo)

sbm = cdll.LoadLibrary("./lib/sbm_c.so")

main = sbm.iterate

main.argtypes = [POINTER(c_float), POINTER(c_float), c_int, c_int, c_int]
# main.restype = c_float

start = time.time()
# energy = main(spin, qubo, dim, window, maxStep)

main(spin, qubo, dim, window, maxStep)
    
end = time.time()


spin = ctplib.as_array(spin)
spin = np.sign(spin)
spin = np.expand_dims(spin,axis=1)
    
qubo = np.reshape(qubo, [dim, dim])

energy = -.5 * (spin.T @ qubo @ spin)[0][0]

print(energy)

spin = spin.T[0].tolist()
for i in range(dim):
    print("+" if spin[i]==1 else "-", sep="", end="")
print("\nspent time: ", end-start)

# # test code
# binary = np.expand_dims(binary, axis=1)
# print( -0.5 * binary.T @ qubo @ binary)
# # test code