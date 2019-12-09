#pragma once
#ifndef PCG_CU
#define PCG_CU
#include <iostream>
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <assert.h>
#include <cmath>
#include <vector>
#include "cublas_v2.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#define N_MAX 4000
#define NNZ_MAX 15248
enum constuctor_type {
	HostArray = cudaMemcpyHostToDevice,
	DeviceArray = cudaMemcpyDeviceToDevice
};

__global__ void initialvalue(int N, double* A, double* B, double* Minverse, double* r, double* z, double* p, int* IA, int* JA)
{
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int tid = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	while (tid < N)
	{
		int jtmp = IA[tid + 1] - IA[tid];
		for (int j = 0; j < jtmp; j++)
		{
			if (JA[j + IA[tid]] == tid)
			{
				Minverse[tid] = 1.0 / A[j + IA[tid]];
			}
		}
		r[tid] = B[tid];
		z[tid] = Minverse[tid] * r[tid];
		p[tid] = z[tid];
		tid += (gridDim.x * blockDim.x) * (gridDim.y * blockDim.y);
	}
}
__global__ void VectorAMUtiplyP(int N, double* A, double* p, double* ap, int* IA, int* JA)
{
	 
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int tid = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	while (tid < N)
	{
		double temp = 0;
		int jtemp;
		jtemp = IA[tid + 1] - IA[tid];
		for (int j = 0; j < jtemp; j++)
		{
			temp += A[j + IA[tid]] * p[JA[j + IA[tid]]];
		}
		ap[tid] = temp;
		tid += (gridDim.x * blockDim.x) * (gridDim.y * blockDim.y);
	}
}
__global__ void inerate_ak(double* zr, double* pap, double* ak)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		*ak = (*zr) / (*pap);
	}
}
__global__ void inerate_x(int N, double* p, double* ak, double* x)
{
	 
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int tid = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	while (tid < N)
	{
		//	printf("%d %f %f %f\n", tid, x[tid], *ak, p[tid]);
		x[tid] = x[tid] + (*ak) * p[tid];
		//	printf("%d %f\n", tid, x[tid]);
		tid += (gridDim.x * blockDim.x) * (gridDim.y * blockDim.y);
	}
}
__global__ void inerate_r(int N, double* ak, double* ap, double* r)
{
	 
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int tid = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	while (tid < N)
	{
		r[tid] = r[tid] - (*ak) * ap[tid];
		tid += (gridDim.x * blockDim.x) * (gridDim.y * blockDim.y);
	}
}
__global__ void inerate_z(int N, double* Minverse, double* r, double* z)
{
	 
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int tid = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	while (tid < N)
	{
		z[tid] = Minverse[tid] * r[tid];
		tid += (gridDim.x * blockDim.x) * (gridDim.y * blockDim.y);
	}
}
__global__ void inerate_p(int N, double* zrnew, double* zr, double* z, double* p)
{
	 
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int tid = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	while (tid < N)
	{
		p[tid] = z[tid] + ((*zrnew) / (*zr)) * p[tid];
		tid += (gridDim.x * blockDim.x) * (gridDim.y * blockDim.y);
	}
}
__global__ void decouple_pos(thrust::pair<int, int>* pos, int* pos_x, int* pos_y, double* value) {
	int index = threadIdx.x;
	pos_x[index] = pos[index].first;
	pos_y[index] = pos[index].second;
//	printf("hes %d %d %d %f\n", index, pos_x[index], pos_y[index], value[index]);
}

class PCG {
private:
	double* A;
	double* B;
	int N, NNZ, * csr;
	cudaEvent_t start, stop;
	double* dev_Minverse, * dev_r, * dev_z, * dev_p;
	double* zr, * dev_zr, * dev_ap, * pap, * dev_pap, * ak, * dev_ak, * x, * dev_x, * zrnew, * dev_zrnew;
	int* pos_x, * pos_y;
	dim3 block;
	dim3 grid;
	cublasStatus_t status;
	cublasHandle_t handle;
	int* IA;
	int* JA;
	void initialize(int Ntemp, int NNZtemp, double* Atemp, double* Btemp, int* IAtemp, int* JAtemp, constuctor_type TYPE);
public:
	PCG() = default;
	PCG(int Ntemp, int NNZtemp, thrust::pair<int, int>* coo, double* hes_val, double* right_hand, constuctor_type TYPE);
	PCG(int Ntemp, int NNZtemp, double* Atemp, double* Btemp, int* IAtemp, int* JAtemp, constuctor_type TYPE);
	double* solve_pcg();
	void update_hes(double* Atemp, double* Btemp, constuctor_type TYPE);
	void readIAandJA(int size_Matrix, int size_nozeronumber, int* IAtemp, int* JAtemp, constuctor_type TYPE);
	~PCG();
};

void PCG::readIAandJA(int size_Matrix, int size_nozeronumber, int* IAtemp, int* JAtemp, constuctor_type TYPE)
{
	cudaMalloc((void**)& IA, sizeof(int) * (size_Matrix + 1));
	cudaMalloc((void**)& JA, sizeof(int) * size_nozeronumber);
	cudaMemcpy(IA, IAtemp, sizeof(int) * (size_Matrix + 1), (cudaMemcpyKind)TYPE);
	cudaMemcpy(JA, JAtemp, sizeof(int) * size_nozeronumber, (cudaMemcpyKind)TYPE);
}

PCG::PCG(int Ntemp, int NNZtemp, thrust::pair<int, int>* coo, double* hes_val, double* right_hand, constuctor_type TYPE) {
	
	cudaMalloc((void**)& csr, sizeof(int) * (Ntemp+1));
	cudaMalloc((void**)& pos_y, sizeof(int) * NNZtemp);
	cudaMalloc((void**)& pos_x, sizeof(int) * NNZtemp);
	thrust::device_ptr<double> dev_data_ptr(hes_val);
	thrust::device_ptr<thrust::pair<int, int>> dev_keys_ptr(coo);
	thrust::sort_by_key(dev_keys_ptr, dev_keys_ptr + NNZtemp, dev_data_ptr);
	decouple_pos << <1, NNZtemp>> > (coo, pos_x, pos_y, hes_val);
	cusparseHandle_t handle;
	cusparseStatus_t status = cusparseCreate(&handle);
	cusparseXcoo2csr(handle, pos_x, NNZtemp, N, csr, CUSPARSE_INDEX_BASE_ZERO);
	initialize(Ntemp, NNZtemp, hes_val, right_hand, csr, pos_y, TYPE);
}

void PCG::initialize(int Ntemp, int NNZtemp, double* Atemp, double* Btemp, int* IAtemp, int* JAtemp, constuctor_type TYPE) { //IAtemp 最后一项必须为总非零点数目
	readIAandJA(Ntemp, NNZtemp, IAtemp, JAtemp, TYPE);
	N = Ntemp;
	NNZ = NNZtemp;
	assert(Ntemp <= N_MAX);
	assert(NNZtemp <= NNZ_MAX);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	block = dim3(32, 32);
	grid = dim3((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

	zr = new double;
	pap = new double;
	ak = new double;
	x = new double[N];
	zrnew = new double;

	cudaMalloc((void**)& A, sizeof(double) * NNZ);
	cudaMalloc((void**)& B, sizeof(double) * N);
	cudaMalloc((void**)& dev_Minverse, sizeof(double) * N);
	cudaMalloc((void**)& dev_r, sizeof(double) * N);
	cudaMalloc((void**)& dev_z, sizeof(double) * N);
	cudaMalloc((void**)& dev_p, sizeof(double) * N);
	cudaMalloc((void**)& dev_zr, sizeof(double));
	cudaMalloc((void**)& dev_ap, sizeof(double) * N);
	cudaMalloc((void**)& dev_pap, sizeof(double));
	cudaMalloc((void**)& dev_ak, sizeof(double));
	cudaMalloc((void**)& dev_x, sizeof(double) * N);
	cudaMalloc((void**)& dev_zrnew, sizeof(double));

	cudaMemcpy(A, Atemp, sizeof(double) * NNZtemp, (cudaMemcpyKind)TYPE);
	cudaMemcpy(B, Btemp, sizeof(double) * N, (cudaMemcpyKind)TYPE);


	initialvalue << <grid, block >> > (N, A, B, dev_Minverse, dev_r, dev_z, dev_p, IA, JA);

	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		std::cout << "CUBLAS对象实例化出错" << std::endl;
		getchar();
	}
	cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
}

PCG::PCG(int Ntemp, int NNZtemp, double* Atemp, double* Btemp, int* IAtemp, int* JAtemp, constuctor_type TYPE) { //IAtemp 最后一项必须为总非零点数目
	initialize(Ntemp, NNZtemp, Atemp, Btemp, IAtemp, JAtemp, TYPE);
}

//PCG::PCG(int Ntemp, int NNZtemp, double* Atemp, double* Btemp, int* IAtemp, int* JAtemp, constuctor_type TYPE) { //IAtemp 最后一项必须为总非零点数目
//	readIAandJA(Ntemp, NNZtemp, IAtemp, JAtemp, TYPE);
//	N = Ntemp;
//	NNZ = NNZtemp;
//	assert(Ntemp <= N_MAX);
//	assert(NNZtemp <= NNZ_MAX);
//
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//	cudaEventRecord(start, 0);
//	block = dim3(32, 32);
//	grid = dim3((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
//
//	zr = new double;
//	pap = new double;
//	ak = new double;
//	x = new double[N];
//	zrnew = new double;
//
//	cudaMalloc((void**)& A, sizeof(double) * NNZ);
//	cudaMalloc((void**)& B, sizeof(double) * N);
//	cudaMalloc((void**)& dev_Minverse, sizeof(double) * N);
//	cudaMalloc((void**)& dev_r, sizeof(double) * N);
//	cudaMalloc((void**)& dev_z, sizeof(double) * N);
//	cudaMalloc((void**)& dev_p, sizeof(double) * N);
//	cudaMalloc((void**)& dev_zr, sizeof(double));
//	cudaMalloc((void**)& dev_ap, sizeof(double) * N);
//	cudaMalloc((void**)& dev_pap, sizeof(double));
//	cudaMalloc((void**)& dev_ak, sizeof(double));
//	cudaMalloc((void**)& dev_x, sizeof(double) * N);
//	cudaMalloc((void**)& dev_zrnew, sizeof(double));
//
//	cudaMemcpy(A, Atemp, sizeof(double) * NNZtemp, (cudaMemcpyKind)TYPE);
//	cudaMemcpy(B, Btemp, sizeof(double) * N, (cudaMemcpyKind)TYPE);
//
//
//	initialvalue << <grid, block >> > (N, A, B, dev_Minverse, dev_r, dev_z, dev_p, IA, JA);
//
//	status = cublasCreate(&handle);
//	if (status != CUBLAS_STATUS_SUCCESS)
//	{
//		std::cout << "CUBLAS对象实例化出错" << std::endl;
//		getchar();
//	}
//}

void PCG::update_hes(double* Atemp, double* Btemp, constuctor_type TYPE) {
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	dim3 block(32, 32);
	dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

	zr = new double;
	pap = new double;
	ak = new double;
	x = new double[N];
	zrnew = new double;

	cudaMalloc((void**)& A, sizeof(double) * NNZ);
	cudaMalloc((void**)& B, sizeof(double) * N);
	cudaMalloc((void**)& dev_Minverse, sizeof(double) * N);
	cudaMalloc((void**)& dev_r, sizeof(double) * N);
	cudaMalloc((void**)& dev_z, sizeof(double) * N);
	cudaMalloc((void**)& dev_p, sizeof(double) * N);
	cudaMalloc((void**)& dev_zr, sizeof(double));
	cudaMalloc((void**)& dev_ap, sizeof(double) * N);
	cudaMalloc((void**)& dev_pap, sizeof(double));
	cudaMalloc((void**)& dev_ak, sizeof(double));
	cudaMalloc((void**)& dev_x, sizeof(double) * N);
	cudaMalloc((void**)& dev_zrnew, sizeof(double));

	cudaMemcpy(A, Atemp, sizeof(double) * NNZ, (cudaMemcpyKind)TYPE);
	cudaMemcpy(B, Btemp, sizeof(double) * N, (cudaMemcpyKind)TYPE);


	initialvalue << <grid, block >> > (N, A, B, dev_Minverse, dev_r, dev_z, dev_p, IA, JA);

	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		std::cout << "CUBLAS对象实例化出错" << std::endl;
		getchar();
	}
}

PCG::~PCG() {
	cudaFree(A);
	cudaFree(B);
	cudaFree(dev_Minverse);
	cudaFree(dev_r);
	cudaFree(dev_z);
	cudaFree(dev_p);
	cudaFree(dev_zr);
	cudaFree(dev_ap);
	cudaFree(dev_pap);
	cudaFree(dev_ak);
	cudaFree(dev_x);
	cudaFree(dev_zrnew);
}
double* PCG::solve_pcg() {
	for (int i = 0; i < N; i++)
	{
		cublasDdot(handle, N, dev_z, 1, dev_r, 1, dev_zr);
		VectorAMUtiplyP << <grid, block >> > (N, A, dev_p, dev_ap, IA, JA);
		cublasDdot(handle, N, dev_ap, 1, dev_p, 1, dev_pap);
		inerate_ak << <grid, block >> > (dev_zr, dev_pap, dev_ak);
		inerate_x << <grid, block >> > (N, dev_p, dev_ak, dev_x);
		inerate_r << <grid, block >> > (N, dev_ak, dev_ap, dev_r);
		inerate_z << <grid, block >> > (N, dev_Minverse, dev_r, dev_z);
		cublasDdot(handle, N, dev_z, 1, dev_r, 1, dev_zrnew);
		cudaMemcpy(zrnew, dev_zrnew, sizeof(double), cudaMemcpyDeviceToHost);
		if (sqrt(*zrnew) < 1.0e-8) break;
		inerate_p << <grid, block >> > (N, dev_zrnew, dev_zr, dev_z, dev_p);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	std::cout << time;
	return dev_x;
}

__global__ void print(int* a) {
	if (threadIdx.x == 0) printf("%d", *a);
}

#endif