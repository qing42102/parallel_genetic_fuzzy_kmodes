# Parallel Genetic Fuzzy k-Modes Clustering Algorithm 

This is based on the paper "A genetic fuzzy k-Modes algorithm for clustering categorical data" (https://www.sciencedirect.com/science/article/pii/S0957417407005957).

This project parallelizes the algorithm using CuPy and mpi4py. Most of the calculations are vectorized using CuPy. mpi4py is used to run multiple groups of chromosomes in parallel. 

The results show significant speedup against the NumPy implementation. 

The branches are organized as follow: 
1. `main` branch: NumPy implmentation. 
2. `parallel` branch: CuPy implementation
3. `MPI` branch: mpi4py + CuPy implementation 

### Running the Code
Run scripts/main.py with the following command for `main` and `parallel` branch.

```
python3 main.py
```

For `MPI` branch, 
```
mpirun -n N python3 main.py
```

Note that the MPI built and used needs to be CUDA-aware. https://www.open-mpi.org/faq/?category=buildcuda

### Reference 
Gan, G., Wu, J., & Yang, Z. (2009). A genetic fuzzy k-Modes algorithm for clustering categorical data. *Expert Systems with Applications*, 36(2), 1615-1620.
