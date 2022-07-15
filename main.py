import numpy as np
import numpy.ctypeslib as ctplib
from ctypes import c_float, c_int, cdll, POINTER
import time

time0 = time.time()

np.random.seed(1)
dim = 40000
window = 0
maxStep = 100
qubo = 2 * np.random.rand(dim, dim).astype(np.float32) - 1
qubo = (qubo + qubo.T) / 2
qubo = qubo.flatten()
np.random.seed()
spin = 2 * np.random.rand(dim).astype(np.float32) - 1

time1 = time.time()
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

sbm = cdll.LoadLibrary("./lib/sbm_cu.so")

main = sbm.iterate

main.argtypes = [POINTER(c_float), POINTER(c_float), c_int, c_int, c_int]
# main.restype = c_float

start = time.time()
# energy = main(spin, qubo, dim, window, maxStep)

main(spin, qubo, dim, window, maxStep)

end = time.time()


# spin = ctplib.as_array(spin)
# spin = np.sign(spin)
# spin = np.expand_dims(spin, axis=1)

# qubo = np.reshape(qubo, [dim, dim])

# # energy = -.5 * (spin.T @ qubo @ spin)[0][0]

# # print(energy)

# spin = spin.T[0].tolist()
# for i in range(dim):
#     print("+" if spin[i] == 1 else ("-" if spin[i] == -1 else 0), sep="", end="")

print("time0 = ",time1-time0)
print("time1 = ",start-time1)
print("\nspent time: ", end-start)

# # # test code
# # binary = np.expand_dims(binary, axis=1)
# # print( - binary.T @ qubo @ binary)
# # # test code
