import numpy as np
import numpy.ctypeslib as ctplib
from ctypes import c_float, c_int, cdll, POINTER
import time

np.random.seed(1)

dim = 15
trotterNum = 4
sweeps = 20

qubo = 2 * np.random.rand(dim, dim).astype(np.float32) - 1
qubo = (qubo + qubo.T) / 2
qubo = qubo.flatten()
spin = np.ones(dim, dtype=np.int32)


spin = ctplib.as_ctypes(spin)
qubo = ctplib.as_ctypes(qubo)

simulatedQA = cdll.LoadLibrary("./lib/sqa.so")

main = simulatedQA.simulatedQA

main.argtypes = [POINTER(c_int), POINTER(c_float), c_int, c_int, c_int]
main.restype = c_float


start = time.time()
energy = main(spin, qubo, dim, trotterNum, sweeps)
end = time.time()

spin = ctplib.as_array(spin)

print(f"%.9f" % energy)
print(spin)
print("spent time: ", end-start)
