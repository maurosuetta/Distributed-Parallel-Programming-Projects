# Distributed and Parallel Programming Projects

This repository presents a collection of High-Performance Computing (HPC) projects that explore different paradigms of parallel and distributed programming. The implemented exercises cover key technologies in the field, including **OpenMP, MPI, CUDA, and OpenACC**.

The main objective is to tackle computationally intensive problems by applying parallelization techniques to reduce execution times and analyze performance scaling on shared-memory (multi-core CPU) and distributed-memory (clusters and GPUs) architectures.

Most of the programs have been executed and evaluated on the **Pirineus3 cluster**, a high-performance computing environment. An output file from the cluster is included with each exercise to allow for the observation and comparison of performance metrics.

### A Note on the Code

The source code files in this repository represent a corrected and improved version compared to the one originally submitted in the reports. The modifications are minor and do not affect the core logic, so the reports remain an excellent guide for understanding the approach and results of each implementation.

## Table of Contents

1.  Project 1: OpenMP (Shared Memory)
      * [Cholesky Factorization](https://www.google.com/search?q=%2311-cholesky-factorization)
      * [Histogram Generation](https://www.google.com/search?q=%2312-histogram-generation)
      * [Argmax Algorithm](https://www.google.com/search?q=%2313-argmax-algorithm)
2.  Project 2: MPI (Message Passing)
      * [Monte Carlo Estimation](https://www.google.com/search?q=%2321-monte-carlo-estimation)
      * Flight Controller Simulator
3.  Project 3: CUDA & OpenACC (GPU Acceleration)
      * [Vector Addition](https://www.google.com/search?q=%2331-vector-addition)
      * [Matrix Multiplication](https://www.google.com/search?q=%2332-matrix-multiplication)
      * [Lagrangian Particle Tracking](https://www.google.com/search?q=%2333-lagrangian-particle-tracking)
4.  Environment and Execution

-----

## Project 1: OpenMP (Shared Memory)

This set of exercises focuses on the parallelization of algorithms on shared-memory systems using the **OpenMP** API. The goal is to exploit loop-level and task-level parallelism on multi-core CPUs.

### 1.1. Cholesky Factorization

**Cholesky factorization** is a decomposition of a symmetric, positive-definite matrix $A$ into the product of a lower triangular matrix $L$ and its transpose $L^T$ (or, equivalently, an upper triangular matrix $U$ and its transpose $U^T$).

  * **Problem:** Implement the factorization $A = LU$ where $U = L^T$. The computation of each element of the matrix $U$ depends on results previously calculated, which introduces data dependencies.
  * **Solution Approach:**
    1.  **Computing U:** The loops for computing the upper triangular matrix U are parallelized. Given the data dependencies, where calculating an element $u\_{ij}$ depends on $u\_{ii}$, careful parallelization strategy is needed. Different `schedule` clauses (static, dynamic, guided) are explored to optimize load balancing.
    2.  **Matrix Multiplication ($B=LU$):** The final verification involves a matrix multiplication. This operation is inherently parallel and is addressed by distributing the calculation of the resulting matrix $B$ among the OpenMP threads.

### 1.2. Histogram Generation

This exercise involves building a histogram from a large set of pseudo-randomly generated values.

  * **Problem:** The main challenge in parallelization is the **race condition** that occurs when multiple threads attempt to update the same histogram bin simultaneously.
  * **Solution Approach:** Four different strategies for managing the critical section and ensuring mutual exclusion are implemented and compared:
    1.  **`critical`:** An `omp critical` directive is used to protect access to the histogram array, ensuring that only one thread can modify it at a time.
    2.  **`atomic`:** `omp atomic` is employed to protect the increment operation of each individual bin.
    3.  **`locks`:** OpenMP locks are used to allow threads updating different bins to work in parallel.
    4.  **`reduction`:** The `reduction` clause of OpenMP is used. Each thread works with a private copy of the histogram, and at the end, OpenMP combines all private histograms into a final one.

### 1.3. Argmax Algorithm

The goal is to find the maximum value in a vector and the position (index) where it is located.

  * **Problem:** Although a simple operation, parallelization requires managing the update of two shared variables (the global maximum and its index) by multiple threads.
  * **Solution Approach:** Two parallelization paradigms with OpenMP are explored:
    1.  **Data Parallelism (`omp for`):** The vector is divided among the threads. Each thread finds the local maximum in its subsection. Finally, the local results are combined to find the global maximum.
    2.  **Task Parallelism (`omp task`):** A recursive "divide and conquer" solution is implemented. Each recursive call for a half of the vector is packaged as a task, creating a task tree that OpenMP manages dynamically.

-----

## Project 2: MPI (Message Passing)

These exercises are designed to solve problems on distributed-memory systems using the **MPI (Message Passing Interface)**. The focus is on domain decomposition and managing communication between processes.

### 2.1. Monte Carlo Estimation

A stochastic method is used to approximate the value of the ratio between the volume of a hypersphere and the hypercube that inscribes it in $d$ dimensions. For $d=2$, this relationship allows for the estimation of $\\pi$.

  * **Problem:** A massive number of random points need to be generated and counted to determine how many fall inside the hypersphere. This task is highly parallel.
  * **Solution Approach:** The strategy is "embarrassingly parallel". The total number of samples is divided among all MPI processes. Each process generates its subset of points independently and calculates a local count. In the end, a collective operation (`MPI_Reduce`) is used to sum the counts from all processes and calculate the final estimation.

### 2.2. Flight Controller Simulator

This project simulates the motion of multiple planes over a 2D grid of cells.

  * **Problem:** The sequential simulation becomes inefficient with a large number of planes. Parallelization requires distributing both the data (planes) and the computation (position updates) among multiple processes. The main challenge is managing planes that cross the boundaries between regions assigned to different processes.
  * **Solution Approach:** **Domain decomposition** is used. The 2D grid is divided into subdomains (tiles), and each MPI process becomes responsible for the cells within one of them. At each time step, each process updates the planes in its domain. Planes that exit a domain must be communicated to the corresponding process. Three communication strategies are implemented:
    1.  **Point-to-Point (`Send`/`Recv`):** Direct communication between processes to exchange plane data.
    2.  **Collective (`All-to-all`):** A collective operation is used to redistribute all planes that have changed domains.
    3.  **Collective with Derived Datatypes:** A custom MPI datatype is defined to pack plane information, optimizing the communication process.

-----

## Project 3: CUDA & OpenACC (GPU Acceleration)

This block of exercises explores programming Graphics Processing Units (GPUs) to accelerate massively parallel computations, using both **CUDA** (a low-level programming model) and **OpenACC** (a high-level, directive-based approach).

### 3.1. Vector Addition

A fundamental operation in linear algebra, $c\_{i}=a\_{i}+b\_{i}$.

  * **Problem:** This simple operation serves as an introduction to GPU programming, covering concepts of memory management (host/device), data transfers, and kernel launches.
  * **Solution Approach:**
    1.  **CUDA:** A kernel is written where each GPU thread computes a single element of the result vector. The code explicitly manages memory allocation on the GPU, data copies (Host-to-Device and Device-to-Host), and the configuration of the thread grid.
    2.  **OpenACC:** A directive-based approach is used. Pragmas are added to the sequential C code's vector addition loop, instructing the compiler to parallelize it and offload it to the GPU. Data transfers are managed using explicit data clauses.

### 3.2. Matrix Multiplication

The operation $C = A \\times B$ for square matrices is implemented.

  * **Problem:** Matrix multiplication is computationally intensive ($O(N^3)$) and its performance on a GPU is highly dependent on efficient memory access.
  * **Solution Approach:** A progression of optimizations is implemented in CUDA:
    1.  **Naive Kernel:** Each thread calculates one element of the matrix C. This version suffers from inefficient global memory accesses.
    2.  **Shared Memory Kernel (Tiling):** The *tiling* technique is implemented. Each thread block loads sub-matrices (tiles) of A and B into the fast shared memory of the GPU. Threads in the block perform partial products by reusing data from shared memory, reducing traffic to global memory.
    3.  **cuBLAS:** NVIDIA's `cuBLAS` library, which provides a highly optimized matrix multiplication implementation, is used to establish a maximum performance baseline.

### 3.3. Lagrangian Particle Tracking

The trajectory of a spray of fuel droplets is simulated, where each particle's position is updated at each time step using the explicit Euler method.

  * **Problem:** The main simulation loop iterates over time and, at each step, it updates the state (position and velocity) of all particles. The calculation for each particle is independent, making it an ideal candidate for GPU acceleration.
  * **Solution Approach:** **OpenACC** is used to parallelize the particle update loop. Two memory management approaches are explored:
    1.  **Unified Memory:** The compiler and CUDA runtime are allowed to automatically manage data transfers between the CPU and GPU.
    2.  **Manual Data Management:** Data directives (`enter data`, `exit data`, `update`) are used to explicitly control when and what data is moved, minimizing transfers and maximizing performance.
        Additionally, **asynchronous operations** (`async`, `wait`) are implemented to overlap computations on the GPU with data transfers between the host and the device.

-----

## Environment and Execution

Each project is self-contained within its respective directory. To compile and run an exercise:

1.  Navigate to the specific exercise folder. For example, to run the Cholesky Decomposition project:
    `cd OpenMP_Cholesky-Histograms-.../"Cholesky Decomposition"`

2.  Use the provided Makefile to compile the code:
    `make`

3.  Execute the program using the job submission scripts (`job.sh`) adapted for the SLURM scheduler, or directly from the command line with `mpirun`, `srun`, or `./executable` as appropriate.

The Makefiles and scripts are configured to load the necessary modules on the cluster (e.g., `gcc`, `openmpi`, `nvhpc`).the program using the job submission scripts (`job.sh`) adapted for the SLURM scheduler, or directly from the command line with `mpirun`, `srun`, or `./executable` as appropriate.

The Makefiles and scripts are configured to load the necessary modules on the cluster (e.g., `gcc`, `openmpi`, `nvhpc`).
