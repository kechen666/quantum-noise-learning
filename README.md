# quantum-noise-learning

## 简介

量子噪声学习模块，主要包含了一些常见的量子噪声学习方法，包括：
* randomized benchmarking: 通过随机运行clifford电路，来估计运行clifford量子门引入的平均噪声参数，具体参考论文[1]。
* interleaved randomized benchmarking: 基于randomized benchmarking的变体，通过插入一个特定的clifford量子门，来实现对于特定量子门去极化参数的估计，具体参考论文[2]。
* pauli noise learning: 利用泡利通道学习的方法，通过在多次运行随机的量子通道，具体参考论文[3].


## 软件架构

* rb: randomized benchmarking模块，由于qiskit中已经实现，这里只包含了rb的使用方法。
* irb: interleaved randomized benchmarking模块，由于qiskit中已经实现，这里包含了irb的使用方法以及irb的准确率验证。
* pnl: pauli noise learning模块，包含了pauli noise learning的实现与使用。

## 实验环境

1.  python
2.  qiskit, qiskit-aer
3.  numpy
4. scipy


## 使用说明

具体参考各个模块的readme文件。

## 参考论文
1.Magesan, Easwar, et al. “Robust Randomized Benchmarking of Quantum Processes.” Physical Review Letters, vol. 106, no. 18, May 2011, p. 180504. arXiv.org, https://doi.org/10.1103/PhysRevLett.106.180504.
2.Magesan, Easwar, et al. “Efficient Measurement of Quantum Gate Error by Interleaved Randomized Benchmarking.” Physical Review Letters, vol. 109, no. 8, Aug. 2012, p. 080505. arXiv.org, https://doi.org/10.1103/PhysRevLett.109.080505.
3.Harper, Robin, et al. “Efficient Learning of Quantum Noise.” Nature Physics, vol. 16, no. 12, Dec. 2020, pp. 1184–88. arXiv.org, https://doi.org/10.1038/s41567-020-0992-8.