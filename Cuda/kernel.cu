#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <memory>
#include <iostream>
#include <iostream>
#include <cmath>
#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <cusparse.h>
#include <cstdlib>
#include "pcg.cuh"
#include "Kernel_Energy.cuh"
//double* x, * devx, * val, * gra, * r, * graMax;
//double* hes_value;
////int size;
//int* pos_x, * pos_y;
//int* csr;
double* x;
//thrust::pair<int, int> *device_pos;
//typedef double (*fp)(double);
//typedef void (*val_fp)(double*, double*, int);
//typedef void (*valsum_fp)(double*, double*,int);
//typedef void (*gra_fp)(double*, double*, int);
//typedef void (*gramin_fp)(double*, double*,int);
//typedef void (*hes_fp)( double*, thrust::pair<int, int>*, double*, int);
//typedef void (*print_fp)(double*, int);
int numSMs;
__global__ void decouple_pos(thrust::pair<int, int>* pos, int* pos_x, int* pos_y, int size) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
		printf("%d %d decouple\n", i, size);
		pos_x[i] = pos[i].first;
		pos_y[i] = pos[i].second;
	}

}




__device__ double sqr(double x) {
	return x * x;
}
__global__ void calculate_pos(double* devx, thrust::pair<int, int>* pos, double* val, int N) {
	for (int index = blockIdx.x * blockDim.x + threadIdx.x;
		index < N;
		index += blockDim.x * gridDim.x)
	{
		int pre = index - 1 == -1 ? N - 1 : index - 1;
		int next = index + 1 == N ? 0 : index + 1;
		pos[3 * index] = thrust::make_pair<int, int>(index, pre);
		pos[3 * index + 1] = thrust::make_pair<int, int>(index, index);
		pos[3 * index + 2] = thrust::make_pair<int, int>(index, next);
		val[3 * index] = sin(2 * devx[index] * devx[pre]) + 2 * devx[index] * devx[pre] * cos(2 * devx[index] * devx[pre]);
		val[3 * index + 1] = 2 * sqr(devx[pre]) * cos(2 * devx[index] * devx[pre]) + 2 * sqr(devx[next]) * cos(2 * devx[index] * devx[next]);
		val[3 * index + 2] = sin(2 * devx[index] * devx[next]) + 2 * devx[index] * devx[next] * cos(2 * devx[index] * devx[next]);;
		printf("hes %d %d %d %f\n", 3 * index, pos[3 * index].first, pos[3 * index].second, val[3 * index]);
		printf("hes %d %d %d %f\n", 3 * index + 1, pos[3 * index + 1].first, pos[3 * index + 1].second, val[3 * index + 1]);
		printf("hes %d %d %d %f\n", 3 * index + 2, pos[3 * index + 2].first, pos[3 * index + 2].second, val[3 * index + 2]);
	}

}
//namespace Kernel_Energy {
//	enum constuctor_type {
//		HostArray = cudaMemcpyHostToDevice,
//		DeviceArray = cudaMemcpyDeviceToDevice
//	};
//}
//
//class kernel_energy {
//public:
//	val_fp val;
//	valsum_fp valsum;
//	gra_fp gra;
//	gramin_fp gramin;
//	hes_fp hes;
//	int size;
//	int numSMs;
//	int hes_val_size;
//	double* x, * dev_x, * dev_val, * dev_gra, * dev_val_sum, * dev_gra_min;
//	double* dev_hes_val;
//	int* dev_pos_x, * dev_pos_y;
//	int* dev_csr_index;
//	double* temp;
//	thrust::pair<int, int>* dev_pos;
//	kernel_energy() = default;
//	kernel_energy(val_fp _val, gra_fp _gra, hes_fp _hes, valsum_fp _valsum, gramin_fp _gramin, int _size, int _hes_val_size, double* init_x = 0) {
//		cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
//		val = _val;
//		gra = _gra;
//		hes = _hes;
//		valsum = _valsum;
//		gramin = _gramin;
//		size = _size;
//		hes_val_size = _hes_val_size;
//		std::cout << size << hes_val_size << std::endl;
//		x = new double[size];
//
//		cudaMalloc((void**)& dev_x, size * sizeof(double));
//		cudaMalloc((void**)& dev_gra, size * sizeof(double));
//		cudaMalloc((void**)& temp, size * sizeof(double));
//		cudaMalloc((void**)& dev_val_sum, sizeof(double));
//		cudaMalloc((void**)& dev_gra_min, sizeof(double));
//		cudaMalloc((void**)& dev_val, size * sizeof(double));
//		cudaMalloc((void**)& dev_pos_x, (hes_val_size + 1) * sizeof(int));
//		cudaMalloc((void**)& dev_pos_y, (hes_val_size + 1) * sizeof(int));
//		cudaMalloc((void**)& dev_csr_index, (size + 1) * sizeof(int));
//		cudaMalloc((void**)& dev_hes_val, hes_val_size * sizeof(double));
//		cudaMemcpy(dev_x, init_x, size * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
//		printf("devx\n");
//		printff << <1, 1 >> > (dev_x, size);
//		cudaMemcpy(x, init_x, size * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToHost);
//		cudaMalloc((void**)& dev_pos, hes_val_size* sizeof(thrust::pair<int, int>));
//	}
//	void calc_val() {
//		val << <32 * numSMs, 256 >> > (dev_x, dev_val, size);
//		cudaThreadSynchronize();
//		valsum << <1,size >> > (dev_val, dev_val_sum,size);
//	}
//
//	void calc_gra() {
//		gra << <32 * numSMs, 256 >> > (dev_x, dev_gra, size);
//		cudaThreadSynchronize();
//	}
//	double max_abs_gra() {
//		double re;
//		cudaMemcpy(temp, dev_gra, size * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
//		gramin << <1,size>> > (temp, dev_gra_min, size);
//		cudaThreadSynchronize();
//		cudaMemcpy(&re, dev_gra_min, sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
//		re = fabs(re);
//		printf("MAX !%f\n", re);
//		return re;
//
//	}
//	void devx_add_vec(double* vec, constuctor_type arrayType) {
//		cudaMemcpy(temp, vec, size*sizeof(double), (cudaMemcpyKind)arrayType);
//		add_vec << <32 * numSMs, 256 >> > (dev_x, temp, size);
//		cudaThreadSynchronize();
//	}
//
//	void calc_hes(int hes_cal_index_size=0) {		//传入生成矩阵的kernel的size大小
//		if (hes_cal_index_size == 0) hes_cal_index_size = size;
//		calculate_pos<< <32 * numSMs, 256 >> > (dev_x, dev_pos, dev_hes_val, hes_cal_index_size);
//		cudaThreadSynchronize();
//		//getchar();
//		thrust::device_ptr<double> dev_data_ptr(dev_hes_val);
//		thrust::device_ptr<thrust::pair<int, int>> dev_keys_ptr(dev_pos);
//		thrust::sort_by_key(dev_keys_ptr, dev_keys_ptr + hes_val_size, dev_data_ptr);
//		decouple_pos << <32 * numSMs, 256 >> > (dev_pos, dev_pos_x, dev_pos_y, hes_val_size);
//		cudaThreadSynchronize();
//		cusparseHandle_t handle;
//		cusparseStatus_t status = cusparseCreate(&handle);
//		cusparseXcoo2csr(handle, dev_pos_x, hes_val_size, size, dev_csr_index, CUSPARSE_INDEX_BASE_ZERO);
//	
//	}
//
//};

__global__ void sum_val(double* val, double* r) {
	int index = threadIdx.x;
	for (int i = 1; i < blockDim.x; i <<= 1) {
		if (index % (i << 1) == i) {
			val[index - i] += val[index];
		}
		__syncthreads();
	}
	if (index == 0) {
		r[0] = val[0];
	}
}


__device__ void wait() {
	for (int i = 1; i <= 10000000; i++);
}


__device__ __host__ inline double Max(double x, double y) {
	x = fabs(x);
	y = fabs(y);
	return x > y ? x : y;
}

__global__ void max_gra(double* gra, double* max) {
	int index = threadIdx.x;
	for (int i = 1; i < blockDim.x; i <<= 1) {
		if (index % (i << 1) == i) {
			gra[index - i] = Max(gra[index - i], gra[index]);
		}
		__syncthreads();
	}
	if (index == 0) {
		max[0] = gra[0];
	}

}

__global__ void calculate_val(double* devx, double* val, int size) {
	for (int index = blockIdx.x * blockDim.x + threadIdx.x;
		index < size;
		index += blockDim.x * gridDim.x)
	{

		int pre = index - 1;
		if (pre < 0) pre += size;
		int next = index + 1;
		if (next >= size) next -= size;
		val[index] = sqr(sin(devx[pre] * devx[index])) * sqr(sin(devx[next] * devx[index]));

	}

	//	wait();
}


__global__ void calculate_gra(double* devx, double* gra,int size) {
	for (int index = blockIdx.x * blockDim.x + threadIdx.x;
		index < size;
		index += blockDim.x * gridDim.x)
	{
		int pre = index - 1;
		if (pre < 0) pre += size;
		int next = index + 1;
		if (next >= size) next -= size;
		gra[index] = devx[pre] * sin(2.0 * devx[index] * devx[pre]) + devx[next] * sin(2.0 * devx[index] * devx[next]);
		printf("gra %d %d %d %f %f %f\n", pre, index, next, sqr(devx[index]), devx[pre] * sin(2.0 * devx[index] * devx[pre]), gra[index]);
	}
}



__global__ void minus_gra(double* gra,int size) {
	for (int index = blockIdx.x * blockDim.x + threadIdx.x;
		index < size;
		index += blockDim.x * gridDim.x)
	{
		gra[index]=0.0-gra[index];
	}
}





__global__ void create_tuple(double* devx, int* pos_x, int* pos_y, double* value, int N) {
	int index = threadIdx.x;
	if (index < N) {
		pos_x[index] = index;
		pos_y[index] = index;
		value[index] = 2 * cosf(2 * devx[index]);
	}
	else if(index == N){
		pos_x[index] = N; 

	}
}
PCG *one;
bool first = true;
double* init_pcg(int N, int NNZ, double* device_As, double* device_Bs, int* device_IAs, int* device_JAs) {
	if (first) {
		one = new PCG(N, NNZ, device_As, device_Bs, device_IAs, device_JAs, DeviceArray);
	}
	else {
		one->update_hes(device_As, device_Bs, DeviceArray);
	}
	return one->solve_pcg();
}

__global__ void print(thrust::pair<int, int>* pos,double*val, int size) {
	for (int i = 0; i < size; i++)
		printf("%d %d %d %f\n", i, pos[i].first, pos[i].second, val[i]);
}


__global__ void printdd(int* pos, int size) {
	for (int i = 0; i < size; i++)
		printf("csr %d\n", pos[i]);
}
//__global__ void decouple_pos(thrust::pair<int, int>* pos, int* pos_x, int* pos_y, double * value, int size) {
//	for (int index = blockIdx.x * blockDim.x + threadIdx.x;
//		index < size;
//		index += blockDim.x * gridDim.x)
//	{
//	pos_x[index] = pos[index].first;
//	pos_y[index] = pos[index].second;	
//	printf("hes %d %d %d %f\n", index, pos_x[index], pos_y[index], value[index]);
//	}
//
//}

int main() {
	using namespace Kernel_Energy;
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
	int size;
	scanf("%d", &size);
	x = new double[size];
	for (int i = 0; i < size; i++) x[i] = i * 1.0 + 10.0;

	kernel_energy test((val_fp)(calculate_val), (gra_fp)calculate_gra, (hes_fp)calculate_pos, (valsum_fp)sum_val, (gramin_fp)max_gra, size, 3 * size, x);
	test.calc_val();
	test.calc_gra();
	double* delta_x;
	double eps = 1e-6;
	while (test.max_abs_gra() > eps) {
		test.calc_hes(size);
		test.calc_gra();
		cudaThreadSynchronize();
		minus_gra << <1, size >> > (test.dev_gra, size);
		cudaThreadSynchronize();
		delta_x = init_pcg(size, 3 * size, test.dev_hes_val, test.dev_gra,test.dev_csr_index, test.dev_pos_y);
		test.devx_add_vec(delta_x, Kernel_Energy::constuctor_type::DeviceArray);
		test.calc_gra();
	}
	
	cudaMemcpy(x, test.dev_x, size* sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; i++) {
		int next = i + 1;
		if (next == size) next = 0;
		printf("x[%d]*x[%d]=%f %f\n", i, next, x[i] * x[next],x[i]);
	}

}