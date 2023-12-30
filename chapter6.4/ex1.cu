#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

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

#define PRINT_FIRST_TWENTY(A)         \
	do                                \
	{                                 \
		for (int i = 0; i < 20; i++)  \
		{                             \
			std::cout << A[i] << " "; \
		}                             \
		std::cout << std::endl;       \
	} while (0)

#define TILE_SIZE 32
// fully general kernel with tiling + corner turning
// assume size has been set correctly
__global__ void gemm_tile_ct(float *A, float *B, float *C, int A_w, int A_h, int B_w, int B_h)
{
	// assert dimensions are compatible
	// init shared memory for tiling
	__shared__ float A_tile[TILE_SIZE][TILE_SIZE];
	__shared__ float B_tile[TILE_SIZE][TILE_SIZE];
	float sum = 0;
	for (int k = 0; k < (A_w + TILE_SIZE - 1) / TILE_SIZE; k++)
	{
		// load A tile
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = k * TILE_SIZE + threadIdx.x;
		if (row < A_h && col < A_w)
			A_tile[threadIdx.y][threadIdx.x] = A[row * A_w + col];
		else
			A_tile[threadIdx.y][threadIdx.x] = 0;

		// load B tile, corner turning to coalesce memory access
		row = k * TILE_SIZE + threadIdx.x;
		col = blockIdx.x * blockDim.x + threadIdx.y;
		if (row < B_h && col < B_w)
			B_tile[threadIdx.x][threadIdx.y] = B[col * B_h + row];
		else
			B_tile[threadIdx.x][threadIdx.y] = 0;
		__syncthreads();
		// if (threadIdx.x == 0 && threadIdx.y == 0)
		// {
		// 	// print A tile
		// 	for (int i = 0; i < 3; i++)
		// 	{
		// 		for (int j = 0; j < 3; j++)
		// 		{
		// 			printf("%f ", A_tile[i][j]);
		// 		}
		// 		printf("\n");
		// 	}
		// 	for (int i = 0; i < 3; i++)
		// 	{
		// 		for (int j = 0; j < 3; j++)
		// 		{
		// 			printf("%f ", B_tile[i][j]);
		// 		}
		// 		printf("\n");
		// 	}
		// }
		for (int i = 0; i < TILE_SIZE; i++)
		{
			sum += A_tile[threadIdx.y][i] * B_tile[i][threadIdx.x];
		}
		__syncthreads();
	}

	// coalesce write back to global memory
	int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
	int globalCol = blockIdx.x * blockDim.x + threadIdx.x;
	if (globalRow < A_h && globalCol < B_w)
		C[globalRow * B_w + globalCol] = sum;
}

// fully general kernel with tiling
// assume size has been set correctly
__global__ void gemm_tile_no_ct(float *A, float *B, float *C, int A_w, int A_h, int B_w, int B_h)
{
	// assert dimensions are compatible
	// init shared memory for tiling
	__shared__ float A_tile[TILE_SIZE][TILE_SIZE];
	__shared__ float B_tile[TILE_SIZE][TILE_SIZE];
	float sum = 0;
	for (int k = 0; k < (A_h + TILE_SIZE - 1) / TILE_SIZE; k++)
	{
		// load A tile
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = k * TILE_SIZE + threadIdx.x;
		if (row < A_h && col < A_w)
			A_tile[threadIdx.y][threadIdx.x] = A[row * A_w + col];
		else
			A_tile[threadIdx.y][threadIdx.x] = 0;

		// load B tile
		row = k * TILE_SIZE + threadIdx.y;
		col = blockIdx.x * blockDim.x + threadIdx.x;
		if (row < B_h && col < B_w)
			B_tile[threadIdx.y][threadIdx.x] = B[col * B_h + row];
		else
			B_tile[threadIdx.y][threadIdx.x] = 0;
		__syncthreads();
		// if (threadIdx.x == 0 && threadIdx.y == 0)
		// {
		// 	// print A tile
		// 	for (int i = 0; i < 3; i++)
		// 	{
		// 		for (int j = 0; j < 3; j++)
		// 		{
		// 			printf("%f ", A_tile[i][j]);
		// 		}
		// 		printf("\n");
		// 	}
		// 	// print B tile
		// 	// print A tile
		// 	for (int i = 0; i < 3; i++)
		// 	{
		// 		for (int j = 0; j < 3; j++)
		// 		{
		// 			printf("%f ", B_tile[i][j]);
		// 		}
		// 		printf("\n");
		// 	}
		// }
		for (int i = 0; i < TILE_SIZE; i++)
		{
			sum += A_tile[threadIdx.y][i] * B_tile[i][threadIdx.x];
		}
		__syncthreads();
	}

	// coalesce write back to global memory
	int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
	int globalCol = blockIdx.x * blockDim.x + threadIdx.x;
	if (globalRow < A_h && globalCol < B_w)
		C[globalRow * B_w + globalCol] = sum;
}

void hostMatMul(float *A, float *B, float *C, int A_w, int A_h, int B_w, int B_h)
{
	for (int r = 0; r < A_h; r++)
	{
		for (int c = 0; c < B_w; c++)
		{
			float sum = 0;
			for (int k = 0; k < A_w; k++)
			{
				sum += A[r * A_w + k] * B[k * B_w + c];
			}
			C[r * B_w + c] = sum;
		}
	}
}

void transpose(float *A, float *A_transpose, int A_w, int A_h)
{
	for (int i = 0; i < A_w; i++)
	{
		for (int j = 0; j < A_h; j++)
		{
			A_transpose[i * A_h + j] = A[j * A_w + i];
		}
	}
}

void initRandomMatrix(float *A, int A_w, int A_h)
{
	for (int i = 0; i < A_w * A_h; i++)
	{
		A[i] = rand() % 2;
	}
}

int main()
{
	float *h_A, *h_B, *h_B_transpose, *h_C, *h_C_ref; // Host matrices
	float *d_A, *d_B, *d_C;							  // Device matrices

	// Allocate memory on host and device, initialize matrices, and set up grid and block dimensions as shown in the complete code example.
	const int A_w = 1234;
	const int A_h = 4567;
	const int B_w = 4567;
	const int B_h = 1022;
	size_t A_bytes = A_w * A_h * sizeof(float);
	size_t B_bytes = B_w * B_h * sizeof(float);
	size_t C_bytes = A_h * B_w * sizeof(float);
	CUDA_CHECK(cudaMallocHost((void **)&h_A, A_bytes));
	CUDA_CHECK(cudaMallocHost((void **)&h_B, B_bytes));
	CUDA_CHECK(cudaMallocHost((void **)&h_B_transpose, B_bytes));
	CUDA_CHECK(cudaMallocHost((void **)&h_C, C_bytes));
	CUDA_CHECK(cudaMallocHost((void **)&h_C_ref, C_bytes));
	CUDA_CHECK(cudaMalloc((void **)&d_A, A_bytes));
	CUDA_CHECK(cudaMalloc((void **)&d_B, B_bytes));
	CUDA_CHECK(cudaMalloc((void **)&d_C, C_bytes));

	// init random matrix
	initRandomMatrix(h_A, A_w, A_h);
	initRandomMatrix(h_B, B_w, B_h);

	std::chrono::high_resolution_clock::time_point start, end;
	double hostDuration, ctKernelDuration, noCtKernelDuration;

	// calculate reference on host
	start = std::chrono::high_resolution_clock::now();
	// transpose(h_B, h_B_transpose, B_w, B_h); // count the transpose time cost
	// hostMatMul(h_A, h_B, h_C_ref, A_w, A_h, B_w, B_h);
	end = std::chrono::high_resolution_clock::now();
	hostDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

	dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
	dim3 blocksPerGrid((B_w + TILE_SIZE - 1) / TILE_SIZE, (A_h + TILE_SIZE - 1) / TILE_SIZE);

	PRINT_FIRST_TWENTY(h_A);
	PRINT_FIRST_TWENTY(h_B_transpose);

	// launch no ct kernel
	// copy matrices from host to device
	start = std::chrono::high_resolution_clock::now();
	CUDA_CHECK(cudaMemcpy(d_A, h_A, A_bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, h_B_transpose, B_bytes, cudaMemcpyHostToDevice));
	gemm_tile_no_ct<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, A_w, A_h, B_w, B_h);
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaMemcpy(h_C, d_C, C_bytes, cudaMemcpyDeviceToHost));
	end = std::chrono::high_resolution_clock::now();
	noCtKernelDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	// print first 20 elements of h_C
	PRINT_FIRST_TWENTY(h_C_ref);

	PRINT_FIRST_TWENTY(h_C);
	if (memcmp(h_C_ref, h_C, C_bytes) != 0)
	{
		std::cerr << "no ct kernel result error\n";
		exit(EXIT_FAILURE);
	}

	// launch ct kernel
	// copy matrices from host to device
	start = std::chrono::high_resolution_clock::now();
	CUDA_CHECK(cudaMemcpy(d_A, h_A, A_bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, h_B_transpose, B_bytes, cudaMemcpyHostToDevice));
	gemm_tile_ct<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, A_w, A_h, B_w, B_h);
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaMemcpy(h_C, d_C, C_bytes, cudaMemcpyDeviceToHost));
	end = std::chrono::high_resolution_clock::now();
	ctKernelDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	// print first 20 elements of h_C
	PRINT_FIRST_TWENTY(h_C_ref);

	PRINT_FIRST_TWENTY(h_C);
	// print first 20 elements of h_C_ref
	if (memcmp(h_C_ref, h_C, C_bytes) != 0)
	{
		std::cerr << "ct kernel result error\n";
		exit(EXIT_FAILURE);
	}

	// print time
	std::cout << "Host computation time: " << hostDuration << " ms\n";
	std::cout << "Corner turning kernel computation time: " << ctKernelDuration << " ms\n";
	std::cout << "No corner turning kernel computation time: " << noCtKernelDuration << " ms\n";

	// free memory
	CUDA_CHECK(cudaFreeHost(h_A));
	CUDA_CHECK(cudaFreeHost(h_B));
	CUDA_CHECK(cudaFreeHost(h_B_transpose));
	CUDA_CHECK(cudaFreeHost(h_C));
	CUDA_CHECK(cudaFreeHost(h_C_ref));
	CUDA_CHECK(cudaFree(d_A));
	CUDA_CHECK(cudaFree(d_B));
	CUDA_CHECK(cudaFree(d_C));

	return 0;
}
