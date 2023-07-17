import numpy as np

import code.utilities.utility as util

from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator


class Oracle:

    def __init__(self, n: int, logic: str):
        """
        Instanciates an object of the Oracle class which is the query oracle
        for the target function.

        Arguments:
            - n: int, the dimension of the input space
            - logic: str, corresponding to the truth table of the target function

        Returns:
            - An object of the class Oracle with attributes:
                * dim: the dimension of the input space
                * logic: the string corresponding to the truth table
                * gate_down: the quantum gate corresponding to the oracle down arrow
                * gate_up: the quantum gate corresponding to the oracle down arrow
                * function_gate: the gate corresponding to the function
        """
        # if len(logic) != 2**n:
        #     raise ValueError(f"The length of gates is {len(logic)}, it should be {2**n}")
        
        self.dim = n
        self.logic = logic

        function_gate = self.__get_function_gate()
        permut_gate = self.__get_permutation_gate()
        amp_down_gate = self.__get_amplitude_gate("down")
        amp_up_gate = self.__get_amplitude_gate("up")

        self.function_gate = function_gate

        qc_down = QuantumCircuit(n+1)
        qc_down.append(amp_down_gate, range(n))
        qc_down.append(permut_gate, range(n))
        qc_down.append(function_gate, range(n+1))

        self.gate_down = qc_down.to_gate(label="Oracle down")

        qc_up = QuantumCircuit(n+1)
        qc_up.append(amp_up_gate, range(n))
        qc_up.append(permut_gate, range(n))
        qc_up.append(function_gate, range(n+1))

        self.gate_up = qc_up.to_gate(label="Oracle up")

    
    def __get_function_gate(self):
        n = self.dim
        uni = np.eye(2**(n+1))

        for i in range(2**n):
            if self.logic[i] == "1":
                bin_i = np.binary_repr(i,n)
                col_num = int(bin_i[::-1],2)
                uni[:, [col_num, col_num+2**n]] = uni[:, [col_num+2**n, col_num]]

        return Operator(uni)
    
    
    def __get_amplitude_gate(self, mode="down"):
        n = self.dim
        qc = QuantumCircuit(n)
        for i in range(n):
            x = 2**(2**(n-1-i))
            if mode == "down":
                angle = 2*np.arccos(np.sqrt(x/(x+1)))
            else:
                angle = 2*np.arcsin(np.sqrt(x/(x+1)))

            qc.ry(angle, i)

        return qc.to_gate()
        

    def __get_permutation_gate(self):
        n = self.dim
        permut_uni = np.zeros((2**n, 2**n))
        permut_array = util.get_permutation(n)

        for i in range(2**n):
            original = int(format(i, f"0{n}b")[::-1],2)
            swap = int(format(permut_array[i], f"0{n}b")[::-1],2)
            permut_uni[swap, original] = 1

        return Operator(permut_uni)

