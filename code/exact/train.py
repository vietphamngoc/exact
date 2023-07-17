from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.providers.aer import QasmSimulator

from code.circuits.oracle import Oracle
from code.circuits.tnn import TNN


simulator = QasmSimulator()

def train_tnn(ora: Oracle, tun_net: TNN, cut: int=100):
    """
    Function performing the exact learning.

    Arguments:
        - ora: Oracle, the query oracle for the target concept
        - tun_net: TNN, the network to be tuned
        - cut: int (default=100), the cut off threshold
        - step: int (default=1), the increment step size

    Returns:
        - The number of updates needed to learn exactly the target function
    """
    n = ora.dim
    N = 2**(2**n)
    n_update = 0
    switch = False
    errors = []
    # Stops when k = k_max and s = 0
    while errors != [] or not switch:
        errors = []
        # Creating the circuit
        qr = QuantumRegister(n, 'x')
        qar = QuantumRegister(1, 'a')
        cr = ClassicalRegister(n)
        car = ClassicalRegister(1)
        qc = QuantumCircuit(qr, qar, cr, car)

        # Applying the oracle and the network
        if not switch:
            qc.append(ora.gate_down, range(n+1))
        else:
            qc.append(ora.gate_up, range(n+1))

        qc.append(tun_net.network, range(n+1))

        # Measuring
        qc.measure(qr, cr)
        qc.measure(qar, car)

        # Running the circuit
        compiled_circuit = transpile(qc, simulator)
        job = simulator.run(compiled_circuit, shots=N)
        result = job.result()
        counts = result.get_counts(compiled_circuit)

        # Getting the errors and corrects and counting the errors
        for sample in counts:
            inpt = sample[2:][::-1]
            if sample[0] == "1":
                errors.append(inpt)

        if errors != []:
            tun_net.update_tnn(errors)
            n_update += 1

        else:
            if not switch:
                switch = True
                errors = [0]
    return n_update