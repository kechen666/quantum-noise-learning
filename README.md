# quantum-noise-learning

## 简介

量子噪声学习模块，主要包含了一些常见的量子噪声学习方法，包括：
* randomized benchmarking: 通过随机运行clifford电路，来估计运行clifford量子门引入的平均噪声参数，具体参考论文[1]。
* interleaved randomized benchmarking: 基于randomized benchmarking的变体，通过插入一个特定的clifford量子门，来实现对于特定量子门去极化参数的估计，具体参考论文[2]。
* pauli noise learning: 利用泡利通道学习的方法，通过在多次运行随机的量子通道，具体参考论文[3].


## 软件架构

* randomized_benchmarking: randomized benchmarking模块，由于qiskit中已经实现，这里只包含了rb的使用方法。
* interleaved_randomized_benchmarking: interleaved randomized benchmarking模块，由于qiskit中已经实现，这里包含了irb的使用方法以及irb的准确率验证。
* pauli_noise_learning: pauli noise learning模块，包含了pauli noise learning的实现与使用。

## 实验环境
qiskit的版本太高，可能在构建噪声模型过程中，无法直接使用qiskit提供的含噪模拟器，推荐与下面版本一致。其他配置的版本暂时没有严格要求。

1. python == 3.8
2. qiskit == 0.45.2, qiskit-aer == 0.13.2, qiskit_experiments == 0.5.4.
3. numpy == 1.24.4
4. scipy == 1.10.1
5. sympy == 1.12


## 使用说明

具体可以参考对应文件中的ipynb文件，提供了较为详细的使用示例。

## 参考论文
1. Magesan, Easwar, et al. “Robust Randomized Benchmarking of Quantum Processes.” Physical Review Letters, vol. 106, no. 18, May 2011, p. 180504. arXiv.org, https://doi.org/10.1103/PhysRevLett.106.180504.
2. Magesan, Easwar, et al. “Efficient Measurement of Quantum Gate Error by Interleaved Randomized Benchmarking.” Physical Review Letters, vol. 109, no. 8, Aug. 2012, p. 080505. arXiv.org, https://doi.org/10.1103/PhysRevLett.109.080505.
3. Harper, Robin, et al. “Efficient Learning of Quantum Noise.” Nature Physics, vol. 16, no. 12, Dec. 2020, pp. 1184–88. arXiv.org, https://doi.org/10.1038/s41567-020-0992-8.