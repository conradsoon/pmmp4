1. Consider matrix addition. Can one use shared memory to reduce the global memory bandwidth consumption? Hint: Analyze the elements that are accessed by each thread and see whether there is any commonality between threads.

Using shared memory won't reduce global memory consumption because there is no computation which requires threads to "share" data. In case of matrix multiplication, multiple threads need to share the same column/row to execute matmul, therefore sharing data reduces hte overall number of global memory r/w required. In case of matrix addition, global memory r/w still on order of O(n^2) in any case.

2. Draw the equivalent of Fig. 5.7 for an 8x8 matrix multiplication with 2x2 tiling and 4x4 tiling. Verify that the reduction in global memory bandwidth is indeed proportional to the dimension size of the tiles.

This is roughly how I proved it.

Assuming a tile width of \(k\), there are \(\left(\frac{N}{k}\right)^2\) tiles in total. Assuming each tile is assigned 1-1 to a block, to calculate this tile \((i,j)\) of the final matrix \(C\), we need to iterate through tiles in \(A\) \((i,k)\) and \((k,j)\) in \(B\). There are \(\frac{N}{k}\) iterations of this, and each iteration requires the loading of \(2k^2\) values. Overall, this gives us \(\left(\frac{N}{k}\right)^2 \cdot \frac{N}{k} \cdot 2k^2\) loads, which is \(2N^3/k\).

3. What type of incorrect execution behavior can happen if one forgot to use one or both \_\_syncthreads() in the kernel of Fig. 5.9?

Read-before-write errors.

Suppose one thread finishes early, then it might move on to the next tile and overwrite the shared memory for that tile location. Then, supposing threads are still working on the previous tile, this might lead to incorrect computation.

4. Assuming that capacity is not an issue for registers or shared memory, give one important reason why it would be valuable to use shared memory instead of registers to hold values fetched from global memory? Explain your answer.

Main reason is because threads in a block can use the same shared memory. In the context of matmul, this allows us to save a factor of N in redundant r/w from global memory. Registers are local only to threads, and as such using registers only will prevent us from sharing memory, forcing us to load on the order of N^3 values from global memory. Any speed advantage from using registers is dwarfed by the I/O time to move from global memory.

5. For our tiled matrix-matrix multiplication kernel, if we use a 32x32 tile, what is the reduction of memory bandwidth usage for input matrices M and N?

Tiling saves memory bandwidth by factor of O(N). So roughly 32x reduction in memory bandwidth usage. (?)

6. Assume that a CUDA kernel is launched with 1000 thread blocks, each of which has 512 threads. If a variable is declared as a local variable in the kernel, how many versions of the variable will be created through the lifetime of the execution of the kernel?

1000 blocks \* 512 threads in each block = 512 000 versions of the variable.

7. In the previous question, if a variable is declared as a shared memory variable, how many versions of the variable will be created through the lifetime of the execution of the kernel?

Shared memory variables are instantiated per-block, so only 1000 versions.

8. Consider performing a matrix multiplication of two input matrices with dimensions N x N. How many times is each element in the input matrices requested from global memory when:
   a. There is no tiling?
   b. Tiles of size T x T are used?

a. N times.
b. N/T times.

9. A kernel performs 36 floating-point operations and seven 32-bit global memory accesses per thread. For each of the following device properties, indicate whether this kernel is compute-bound or memory-bound:
   a. Peak FLOPS = 200 GFLOPS, peak memory bandwidth = 100 GB/second
   b. Peak FLOPS = 300 GFLOPS, peak memory bandwidth = 250 GB/second

36/28 FLOPs per byte accessed.
200GB/(36/28) ~= 163GB/s of memory bandwidth needed to sustain peak FLOPs, therefore memory-bound
300GB/(36/28) ~= 233GB/s of memory bandwith needed to sustain peak FLOPs, therefore compute-bound.

10. To manipulate tiles, a new CUDA programmer has written a device kernel that will transpose each tile in a matrix. The tiles are of size BLOCK_WIDTH by BLOCK_WIDTH, and each of the dimensions of matrix A is known to be a multiple of BLOCK_WIDTH. The kernel invocation and code are shown below. BLOCK_WIDTH is known at compile time and could be set anywhere from 1 to 20.
    a. Out of the possible range of values for BLOCK_SIZE, for what values of BLOCK_SIZE will this kernel function execute correctly on the device?
    b. If the code does not execute correctly for all BLOCK_SIZE values, what is the root cause of this incorrect execution behavior? Suggest a fix to the code to make it work for all BLOCK_SIZE values.

11. Consider the following CUDA kernel and the corresponding host function that calls it:
    a. How many versions of the variable i are there?
    b. How many versions of the array x[] are there?
    c. How many versions of the variable y_s are there?
    d. How many versions of the array b_s[] are there?
    e. What is the amount of shared memory used per block (in bytes)?
    f. What is the floating-point to global memory access ratio of the kernel (in OP/B)?

12. Consider a GPU with the following hardware limits: 2048 threads/SM, 32 blocks/SM, 64K (65,536) registers/SM, and 96 KB of shared memory/SM. For each of the following kernel characteristics, specify whether the kernel can achieve full occupancy. If not, specify the limiting factor:
    a. The kernel uses 64 threads/block, 27 registers/thread, and 4 KB of shared memory/SM.
    b. The kernel uses 256 threads/block, 31 registers/thread, and 8 KB of shared memory/SM.
