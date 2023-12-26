#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define MATRIX_SIZE 4096 // Change this to your desired matrix size
#define CUDA_CHECK(call)                                                                                                    \
	do                                                                                                                      \
	{                                                                                                                       \
		cudaError_t error = call;                                                                                           \
		if (error != cudaSuccess)                                                                                           \
		{                                                                                                                   \
			std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(error) << std::endl; \
			exit(EXIT_FAILURE);                                                                                             \
		}                                                                                                                   \
	} while (0)

// kernel to produce one output matrix row per thread
__global__ void matrixRowMultKernel(float *A, float *B, float *C, int matrixSize)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < matrixSize)
	{
		for (int c = 0; c < matrixSize; c++)
		{
			float sum = 0;
			for (int k = 0; k < matrixSize; k++)
			{
				sum += A[row * matrixSize + k] * B[k * matrixSize + c];
			}
			C[row * matrixSize + c] = sum;
		}
	}
}

// Kernel to produce one output matrix column per thread
__global__ void matrixColMultKernel(float *A, float *B, float *C, int matrixSize)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < matrixSize)
	{
		for (int r = 0; r < matrixSize; r++)
		{
			float sum = 0;
			for (int k = 0; k < matrixSize; k++)
			{
				sum += A[r * matrixSize + k] * B[k * matrixSize + col];
			}
			C[r * matrixSize + col] = sum;
		}
	}
}

void hostMatMul(float *A, float *B, float *C, int matrixSize)
{
	for (int r = 0; r < matrixSize; r++)
	{
		for (int c = 0; c < matrixSize; c++)
		{
			float sum = 0;
			for (int k = 0; k < matrixSize; k++)
			{
				sum += A[r * matrixSize + k] * B[k * matrixSize + c];
			}
			C[r * matrixSize + c] = sum;
		}
	}
}

void initRandomMatrix(float *matrix, int matrixSize)
{
	for (int i = 0; i < matrixSize * matrixSize; i++)
	{
		matrix[i] = rand() % 100;
	}
}

void compareMatrices(float *A, float *B, int matrixSize)
{
	for (int i = 0; i < matrixSize * matrixSize; i++)
	{
		if (A[i] != B[i])
		{
			std::cerr << "Matrix comparison failed at element " << i << std::endl;
			exit(EXIT_FAILURE);
		}
	}
}

int main()
{
	float *h_A, *h_B, *h_C, *h_C_ref; // Host matrices
	float *d_A, *d_B, *d_C;			  // Device matrices

	// Allocate memory on host and device, initialize matrices, and set up grid and block dimensions as shown in the complete code example.
	int matrixBytes = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
	CUDA_CHECK(cudaMallocHost((void **)&h_A, matrixBytes));
	CUDA_CHECK(cudaMallocHost((void **)&h_B, matrixBytes));
	CUDA_CHECK(cudaMallocHost((void **)&h_C, matrixBytes));
	CUDA_CHECK(cudaMallocHost((void **)&h_C_ref, matrixBytes));
	CUDA_CHECK(cudaMalloc((void **)&d_A, matrixBytes));
	CUDA_CHECK(cudaMalloc((void **)&d_B, matrixBytes));
	CUDA_CHECK(cudaMalloc((void **)&d_C, matrixBytes));

	// init random matrix
	initRandomMatrix(h_A, MATRIX_SIZE);
	initRandomMatrix(h_B, MATRIX_SIZE);

	// copy matrices from host to device
	CUDA_CHECK(cudaMemcpy(d_A, h_A, matrixBytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, h_B, matrixBytes, cudaMemcpyHostToDevice));

	// inits
	int threadsPerBlock = 256;
	int blocksPerGrid = (MATRIX_SIZE + threadsPerBlock - 1) / threadsPerBlock;
	std::chrono::high_resolution_clock::time_point start, end;
	double hostDuration, rowKernelDuration, colKernelDuration;

	// calculate reference on host
	start = std::chrono::high_resolution_clock::now();
	hostMatMul(h_A, h_B, h_C_ref, MATRIX_SIZE);
	end = std::chrono::high_resolution_clock::now();
	hostDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

	// launch row kernel
	start = std::chrono::high_resolution_clock::now();
	matrixRowMultKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, MATRIX_SIZE);
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaMemcpy(h_C, d_C, matrixBytes, cudaMemcpyDeviceToHost));
	end = std::chrono::high_resolution_clock::now();
	rowKernelDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	compareMatrices(h_C, h_C_ref, MATRIX_SIZE);

	// launch column kernel
	start = std::chrono::high_resolution_clock::now();
	matrixColMultKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, MATRIX_SIZE);
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaMemcpy(h_C, d_C, matrixBytes, cudaMemcpyDeviceToHost));
	end = std::chrono::high_resolution_clock::now();
	colKernelDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	compareMatrices(h_C, h_C_ref, MATRIX_SIZE);

	// print time
	std::cout << "Host computation time: " << hostDuration << " ms\n";
	std::cout << "Row-wise kernel computation time: " << rowKernelDuration << " ms\n";
	std::cout << "Column-wise kernel computation time: " << colKernelDuration << " ms\n";

	return 0;
}
