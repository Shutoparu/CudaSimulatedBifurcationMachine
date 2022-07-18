import numpy as np
import numpy.ctypeslib as ctplib
from ctypes import c_float, c_int, cdll, POINTER
import time


class SBM:

    '''
    Attributes:
    qubo : np.ndarray
        the qubo matrix in 2D
    init_spin : np.ndarray
        the initial spin in 1D
    maxStep : int
        the maximum steps for the algorithm
    window : int
        the usage of sampling method
    dim : int
        the dimention of the spin array
    time : float
        the time spent on the last execution of the algorithm. default value 0 is set
    '''

    def __init__(
        self,
        qubo: np.ndarray = np.array([[0, 1], [1, 0]]),
        spin: np.ndarray = None,
        maxStep: int = 10000,
        window: int = 0
    ) -> None:
        '''
        Parameters:
        qubo : np.ndarray
            the qubo matrix in 2D. elements will be parsed to np.float32 which is equivalent to "float" in C. default qubo matrix [[0,1],[1,0]] is used.
        init_spin : np.ndarray | None
            the initial spin in 1D with values between {-1,1}. elements will be parsed to np.float32 which is equivalent to "float" in C. if none then a random initial spin is generated
        maxStep : int
            the maximum steps for the algorithm. default value 10,000 is used
        window : int
            the usage of sampling method. default value 0 is used, which disables the use of sampling method.
        '''

        self.qubo = qubo.astype(np.float32)
        self.maxStep = maxStep
        self.window = window

        if np.shape(self.qubo)[0] != np.shape(self.qubo)[1]:
            print("qubo is not a square matrix")
            exit(-1)
        self.dim = np.shape(self.qubo)[0]

        if(spin is None):
            self.spin = 2 * np.random.rand(self.dim).astype(np.float32) -1
        else:
            self.spin = spin.astype(np.float32)
        
        if np.shape(self.qubo)[0] != np.shape(self.spin)[0]:
            print("qubo dimention and spin dimention mismatch")
            exit(-1)
        self.time = 0

    def run(self) -> None:

        spin = ctplib.as_ctypes(self.spin)
        qubo = ctplib.as_ctypes(self.qubo.flatten())

        sbm = cdll.LoadLibrary("./lib/sbm_cu.so")

        main = sbm.iterate

        main.argtypes = [POINTER(c_float), POINTER(
            c_float), c_int, c_int, c_int]

        start = time.time()

        main(spin, qubo, self.dim, self.window, self.maxStep)

        end = time.time()

        self.time = end-start

        spin = ctplib.as_array(spin)
        self.spin = np.sign(spin)

    def getSpinSign(self) -> str:
        spin = np.expand_dims(self.spin, axis=1).T[0].tolist()
        returnString = ""
        for i in range(self.dim):
            returnString += ("+" if spin[i] ==
                             1 else ("-" if spin[i] == -1 else "#"))
        return returnString

    def isingEnergy(self) -> float:
        spin = np.expand_dims(self.spin, axis=1)
        return -.5* (spin.T @ self.qubo @ spin)[0][0]


if __name__ == '__main__':

    np.random.seed(1)
    dim = 10000
    window = 0
    maxStep = 1000
    qubo = 2 * np.random.rand(dim, dim).astype(np.float32) - 1
    qubo = (qubo + qubo.T) / 2
    spin = 2 * np.random.rand(dim).astype(np.float32) - 1

    sbm = SBM(qubo, spin, maxStep, window)
    sbm.run()
    print(sbm.time)
    print(sbm.isingEnergy())
