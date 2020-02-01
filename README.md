# matmul-avx
This implements matrix multiplication with AVX instruction and cache blocking.

Below is the performance of this code. On the same machine, OpenBLAS achieves roughly 30Gflop/s.

| Input dimension | Gflop/s |
| --------------- | ------- |
| 32              | 10.7    |
| 96              | 21.1    |
| 192             | 22.3    |
| 288             | 22.4    |
| 308             | 21.5    |
| 511             | 21.9    |
| 512             | 21.1    |
| 600             | 22.9    |
| 718             | 22.8    |
| 719             | 23.3    |
| 1024            | 23.2    |
| 1025            | 22.9    |
| 1600            | 22.7    |
| 1608            | 23.2    |
| 1747            | 22.8    |
| 1752            | 23.2    |
| 1910            | 23.1    |
| 1920            | 23.6    |
| 2047            | 23.1    |
| 2048            | 23.2    |

## Code of CSE 260

Starter Code for the Matrix Multiplication assignment<br />
Original code provided by Jim Demmel<br />
http://www.cs.berkeley.edu/~knight/cs267/hw1.html<br />
with some modifications by Scott B. Baden at UC San Diego<br />
with some modifications by Bryan Chin at UC San Diego<br />

