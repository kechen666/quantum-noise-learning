
import numpy as np
from numpy.random import Generator, default_rng
from typing import  Optional, Union,List

from qiskit.result import Counts
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Clifford
from qiskit_experiments.library.randomized_benchmarking import CliffordUtils
from qiskit import transpile

from scipy.optimize import curve_fit


def random_pauli_benchmarking(num_qubits: int, rep_m_list: List[int], num_1_qubit_twirl: int, num_2_qubit_twirl: int, shots: int, sim):
    """ This function is use to learning about pauli noise.
        .num_qubits: For each circuit, qubit's number.
        .rep_m_list: the test m list.
        .num_1_qubit_twirl:  For each m, the one clifford circuit by sample.
        .num_2_qubit_twirl:  For each m, the two clifford circuit by sample.
        .shots: For each circuit, repeat run circuit number.
        .sim: simulator for run circuit.
    """
    # For each m
    statistics_counts_list = []
    circuit_list = []
    result_list = []
    for m in rep_m_list:
        circuit_m_list = []
        result_m_list = []
        # counts_m_list = []
        counts_m_dict = dict()
        for job_id in range(num_1_qubit_twirl + num_2_qubit_twirl):
            # initial circuit
            qr = QuantumRegister(num_qubits, 'q')
            circ = QuantumCircuit(qr)
            if job_id < num_1_qubit_twirl:
                circ = random_clifford_circuit(circ, num_qubits, m, 1)
            elif (job_id >= num_1_qubit_twirl) and (job_id < num_1_qubit_twirl+num_2_qubit_twirl):
                circ = random_clifford_circuit(circ, num_qubits, m, 2)
            
            # Apply inverse
            inverse_circ = Clifford(circ.inverse()).to_circuit()
            # compose circuit
            circ_compose = circ.compose(inverse_circ)
            
            transpile_qc = transpile(circ_compose, sim, optimization_level = 0)
            
            # measurement
            transpile_qc.measure_all()
            
            # run in a simulator and get result 
            result = sim.run(transpile_qc, shots = shots).result()
            counts = result.get_counts()
            
            # append result
            circuit_m_list.append(transpile_qc)
            result_m_list.append(result)
            
            # gather the counts about a kind of m
            for key, value in counts.items():
                counts_m_dict[key] = counts_m_dict.get(key, 0) + value
        print(f"now is running m = {m} experement")
        
        # use int to expect the key
        qiskit_counts = Counts(counts_m_dict)
        int_outcomes = qiskit_counts.int_outcomes()
        
        statistics_counts_list.append(int_outcomes)
        circuit_list.append(circuit_m_list)
        result_list.append(result_m_list)
    
    # get the Probability distributions list
    
    return circuit_list, result_list, statistics_counts_list

def random_clifford_circuit(circ: QuantumCircuit, num_qubits: int, rep_m: int, num_qubit_twirl: int=1):
    # clifford_circuit 1 or clifford_circuit 2
    if num_qubit_twirl == 1:
        for _ in range(rep_m):
            # random clifford 1 gate on each qubit
            circ = add_random_one_layer_clifford_gate(circ, num_qubits, num_qubits)
    elif num_qubit_twirl == 2:
        for _ in range(rep_m):
            # random clifford 2 gate on two qubit pair
            circ = add_random_one_layer_clifford_2_qubit_gate(circ,num_qubits, num_qubits//2)
    else:
        raise("no support num_qubit_twirl than 2")
    return circ

def add_random_one_layer_clifford_gate(circ: QuantumCircuit, num_qubits: int, size: int, rng: Optional[Union[int, Generator]] = None):
    if rng is None:
        rng = default_rng()
    elif isinstance(rng, int):
        rng = default_rng(rng)
    # have some speed than random_clifford.
    samples = rng.integers(CliffordUtils.NUM_CLIFFORD_1_QUBIT, size=size)
    
    list_circuit = [CliffordUtils.clifford_1_qubit(i).to_instruction() for i in samples]
    
    for i in range(num_qubits):
        circ.append(list_circuit[i], [i])
    circ.barrier()
    return circ

def add_random_one_layer_clifford_2_qubit_gate(circ: QuantumCircuit, num_qubits: int, size: int, rng: Optional[Union[int, Generator]] = None):
    if rng is None:
        rng = default_rng()
    elif isinstance(rng, int):
        rng = default_rng(rng)
    # have some speed than random_clifford.
    
    samples = rng.integers(CliffordUtils.NUM_CLIFFORD_2_QUBIT, size=size)
    
    list_circuit = [CliffordUtils.clifford_2_qubit_circuit(i).to_instruction() for i in samples]
    
    for i in range(size):
        circ.append(list_circuit[i], [i*2,i*2+1])
    circ.barrier()
    return circ

def iwfht(x):
    """逆快速 Walsh Hadamard 变换函数"""
    n = len(x)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x[j], x[j + h] = x[j] + x[j + h], x[j] - x[j + h]
        h *= 2
    return x

# 指数函数模型，用于拟合
def exponential_model(x, a, b):
    return a * (b ** x)

def fit_iws_lambda(list_m, statistics_counts_list: List[dict], num_qubits: int, num_samples: int, shots: int):
    ws_results = []
    for m_id in range(len(list_m)):
        m = list_m[m_id]
        list_pro = [statistics_counts_list[m_id].get(i, 0) for i in range(2**num_qubits)]
        # print(list_pro)
        # print(sum(list_pro))
        num_exper = int(num_samples*shots)
        ws_result = np.array(iwfht(list_pro))/num_exper
        # print(ws_result)
        ws_results.append(ws_result)
    ws_results = np.array(ws_results)

    lambda_list = []
    for i in range(2**num_qubits):
        x_data = np.array(list_m)
        y_data = ws_results[:,i]
        # 拟合数据
        popt, pcov = curve_fit(exponential_model, x_data, y_data, p0=[1.0, 1.0])
        lambda_list.append(popt[1])

    return lambda_list

def eigenvalue_list_to_pauli_probability(lst):
    # 将实数列表转换为概率值
    probabilities = np.array(lst, dtype=np.float64)
    # make sure is >=0
    if np.min(probabilities) < 0:
        probabilities = probabilities - np.min(probabilities)
        
    # 归一化概率值
    sum_probabilities = np.sum(probabilities)
    
    if sum_probabilities != 0:
        probabilities = probabilities / sum_probabilities    
    return probabilities

# 二进制表示
def to_binary_string(number, length: int):
    return format(number, '0' + str(length) + 'b')

def pauli_probability_to_qubit_error_rate(num_qubits, probabilities):
    qubit_error_rates = []
    for index in range(num_qubits-1, -1,-1):
        qubit_error_rate = 0
        # for qubit
        for i in range(2**num_qubits):
            # for seq
            if to_binary_string(i, num_qubits)[index] == "1":
                qubit_error_rate += probabilities[i]
        qubit_error_rates.append(qubit_error_rate)
    return qubit_error_rates