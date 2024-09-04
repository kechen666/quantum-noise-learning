from qiskit_experiments.library import InterleavedRB
import qiskit.circuit.library as circuits

from qiskit_aer.noise import NoiseModel, depolarizing_error

from qiskit_aer import AerSimulator

from qiskit.quantum_info import process_fidelity
from qiskit import QuantumCircuit

    # from qiskit_aer.noise import depolarizing_error

def interleaved_rb_QEC(backend, lengths, num_samples, seed):
    """利用交叉随机基准测试方法, 获取surface code中所涉及到CNOT门、H门、I门的噪声参数.

    Args:
        backend : Qiskit含噪后端
        lengths (list[int]): 电路深度序列
        num_samples (int): 采样次数
        seed (int): 随机种子

    Returns:
        H门、I门、CNOT门以及相互连接的量子比特对
    """
    num_qubits = backend.num_qubits
    
    # get couple map for cnot
    coordinate_pairs = backend.coupling_map.get_edges()
    unique_pairs = []
    for pair in coordinate_pairs:
        # 检查反转的边是否已经存在
        if (pair[1], pair[0]) not in unique_pairs:
            # 如果不存在，将边添加到集合中
            unique_pairs.append(pair)
    
    # for H gate
    h_gate_error = {}
    id_gate_error = {}
    cx_gate_error = {}
    
    qec_basis_gate = ['h', 'id', 'cx']
    
    for gate in qec_basis_gate:
        # print(f"{gate}")
        if gate == 'h':
            for qubit_id in range(num_qubits):
                qubits = (qubit_id,)
                int_exp2 = InterleavedRB(
                    circuits.HGate(), qubits, lengths, num_samples=num_samples, seed=seed)
                int_expdata2 = int_exp2.run(backend).block_for_results()
                int_results2 = int_expdata2.analysis_results()
                h_gate_error[qubit_id] = int_results2[2].value
                # print(f"{qubit_id}_")
        elif gate == 'id':
            for qubit_id in range(num_qubits):
                qubits = (qubit_id,)
                int_exp2 = InterleavedRB(circuits.IGate(), qubits, lengths, num_samples=num_samples, seed=seed)
                int_expdata2 = int_exp2.run(backend).block_for_results()
                int_results2 = int_expdata2.analysis_results()
                id_gate_error[qubit_id] = int_results2[2].value
                # print(f"{qubit_id}_")
        elif gate == 'cx':
            for qubit_pairs in unique_pairs:
                int_exp2 = InterleavedRB(circuits.CXGate(), qubit_pairs, lengths, num_samples=num_samples, seed=seed)
                int_expdata2 = int_exp2.run(backend).block_for_results()
                int_results2 = int_expdata2.analysis_results()
                cx_gate_error[qubit_pairs] = int_results2[2].value
                # print(f"{qubit_pairs}_")
                
    return h_gate_error, id_gate_error, cx_gate_error, unique_pairs

def get_surface_code_noise_model(h_gate_error, id_gate_error, cx_gate_error, backend, backend_noise_model):
    """构建噪声模型

    Args:
        h_gate_error (dict): 各量子比特对应的H门对应的去极化参数
        id_gate_error (dict): 各量子比特对应的I门对应的去极化参数
        cx_gate_error (dict): 两量子比特对对应的CNOT门对应的去极化参数
        backend : 含噪后端, 用于统一readout error.
        backend_noise_model : 噪声模型, 用于统一readout error.

    Returns:
        含噪模拟后端, 噪声模型
    """
    # Create an empty noise model
    noise_model = NoiseModel()

    # consider down bound - (int_results2[2].value.std_dev)

    # Add H and id depolarizing error to qubit
    # max_param to make sure param is probability. Detail in depolarizing_error.
    for qubit_id, depolarizing_param in h_gate_error.items():
        num_terms = 4**1
        max_param = num_terms / (num_terms - 1)
        dep_error = depolarizing_error((1-depolarizing_param.nominal_value)*max_param, 1)
        noise_model.add_quantum_error(dep_error, ['h'], [qubit_id,])
    for qubit_id, depolarizing_param in id_gate_error.items():
        num_terms = 4**1
        max_param = num_terms / (num_terms - 1)
        dep_error = depolarizing_error((1-depolarizing_param.nominal_value)*max_param, 1)
        noise_model.add_quantum_error(dep_error, ['id'], [qubit_id,])
    for qubit_pairs, depolarizing_param in cx_gate_error.items():
        num_terms = 4**2
        max_param = num_terms / (num_terms - 1)
        dep_error = depolarizing_error((1-depolarizing_param.nominal_value)*max_param, 2)
        # Because the error about IRB, so we consider the cx error about (0,1) and (1,0) is same.
        noise_model.add_quantum_error(dep_error, ['cx'], qubit_pairs)
        noise_model.add_quantum_error(dep_error, ['cx'], (qubit_pairs[1], qubit_pairs[0]))
        
    # add readout error and reset error
    # NoiseModel.from_backend(backend)._local_readout_errors[(0,)]
    for i in range(backend.num_qubits):
        noise_model.add_readout_error(backend_noise_model._local_readout_errors[(i,)], [i,])
        # noise_model.add_quantum_error(backend_noise_model._local_quantum_errors['reset'][(i,)], ['reset'], [i,])

    sim_noise = AerSimulator(noise_model = noise_model)
    return sim_noise, noise_model

def i_gate_fidelity(id_gate_error, id_gate_error_no_thermal_relaxation, backend_noise_model, backend_noise_model_no_thermal_relaxation):
    """评估I门的交叉随机测试学习准确率

    Args:
        id_gate_error (_type_): 交叉随机测试学习到的去极化参数, 包含thermal_relaxation噪声。
        id_gate_error_no_thermal_relaxation (_type_): 交叉随机测试学习到的去极化参数, 包含thermal_relaxation噪声。
        backend_noise_model (_type_): _description_
        backend_noise_model_no_thermal_relaxation (_type_): _description_
    """


    num_qubits = 1

    num_terms = 4**num_qubits
    max_param = num_terms / (num_terms - 1)

    exit_thermal_relaxation_learning = depolarizing_error((1-id_gate_error[0].nominal_value)*max_param, 1).to_quantumchannel()
    no_thermal_relaxation_learning = depolarizing_error((1-id_gate_error_no_thermal_relaxation[0].nominal_value)*max_param, 1).to_quantumchannel()

    exit_thermal_relaxation_real = backend_noise_model._local_quantum_errors['id'][(0,)].to_quantumchannel()
    no_thermal_relaxation_real = backend_noise_model_no_thermal_relaxation._local_quantum_errors['id'][(0,)].to_quantumchannel()

    # 计算量子通道之间的过程保真度

    print("考虑thermal_relaxiaiton噪声, 实际和学习到的噪声模型之间的噪声通道过程保真度:", process_fidelity(exit_thermal_relaxation_learning, exit_thermal_relaxation_real))
    print("不考虑thermal_relaxiaiton噪声, 实际和学习到的噪声模型之间的噪声通道过程保真度:", process_fidelity(no_thermal_relaxation_learning, no_thermal_relaxation_real))

def cnot_gate_process_fidelity(cx_gate_error, cx_gate_error_no_thermal_relaxation, backend_noise_model, backend_noise_model_no_thermal_relaxation):
    num_qubits = 2

    num_terms = 4**num_qubits
    max_param = num_terms / (num_terms - 1)

    exit_thermal_relaxation_learning = depolarizing_error((1-cx_gate_error[(0,1)].nominal_value)*max_param, 2).to_quantumchannel()
    no_thermal_relaxation_learning = depolarizing_error((1-cx_gate_error_no_thermal_relaxation[(0,1)].nominal_value)*max_param, 2).to_quantumchannel()

    exit_thermal_relaxation_real = backend_noise_model._local_quantum_errors['cx'][(0,1)].to_quantumchannel()
    no_thermal_relaxation_real = backend_noise_model_no_thermal_relaxation._local_quantum_errors['cx'][(0,1)].to_quantumchannel()

    # 计算量子通道之间的过程保真度

    print("考虑thermal_relaxiaiton噪声, 实际和学习到的噪声模型之间的噪声通道过程保真度:", process_fidelity(exit_thermal_relaxation_learning, exit_thermal_relaxation_real))
    print("不考虑thermal_relaxiaiton噪声, 实际和学习到的噪声模型之间的噪声通道过程保真度:", process_fidelity(no_thermal_relaxation_learning, no_thermal_relaxation_real))

def h_gate_hellinger_fidelity(sim_noise, backend):
    n_qubits = 1
    circ = QuantumCircuit(n_qubits)

    # Test Circuit
    for _ in range(20):
        circ.h(0)
    circ.measure_all()
    counts1_sum = {}
    counts2_sum = {}
    for i in range(20):
        counts1 = sim_noise.run(circ,shots = 1024).result().get_counts()
        counts2 = backend.run(circ,shots = 1024).result().get_counts()
        for key, value in counts1.items():
            counts1_sum[key] = counts1_sum.get(key, 0) + value
        for key, value in counts2.items():
            counts2_sum[key] = counts2_sum.get(key, 0) + value

    from qiskit.quantum_info import hellinger_distance, hellinger_fidelity
    print(f"学习噪声模型结果:{counts1_sum}, 实际噪声模型结果:{counts2_sum}")
    hellinger_dis = hellinger_distance(counts1_sum, counts2_sum)

    hellinger_fid = hellinger_fidelity(counts1_sum, counts2_sum)
    print(f"hellinger距离:{hellinger_dis}, hellinger保真度:{hellinger_fid}")