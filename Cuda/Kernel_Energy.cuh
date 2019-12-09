#pragma once
#ifndef Kernel_Energy_CUH
#define Kernel_Energy_CUH
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
namespace Kernel_Energy {
	typedef double (*fp)(double);
	typedef void (*val_fp)(double*, double*, int);
	typedef void (*valsum_fp)(double*, double*, int);
	typedef void (*gra_fp)(double*, double*, int);
	typedef void (*gramin_fp)(double*, double*, int);
	typedef void (*hes_fp)(double*, thrust::pair<int, int>*, double*, int);
	typedef void (*print_fp)(double*, int);
	__global__ void decouple_pos(thrust::pair<int, int>* pos, int* pos_x, int* pos_y, int size) {
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
			pos_x[i] = pos[i].first;
			pos_y[i] = pos[i].second;
		}

	}

	__global__ void add_vec(double* sum, double* addA, int size) {
		for (int index = blockIdx.x * blockDim.x + threadIdx.x;
			index < size;
			index += blockDim.x * gridDim.x)
		{
			sum[index] += addA[index];
		}

	}


	enum constuctor_type {
		HostArray = cudaMemcpyHostToDevice,
		DeviceArray = cudaMemcpyDeviceToDevice
	};

	class kernel_energy {
	public:
		val_fp val;
		valsum_fp valsum;
		gra_fp gra;
		gramin_fp gramin;
		hes_fp hes;
		int size;
		int numSMs;
		int hes_val_size;
		double* x, * dev_x, * dev_val, * dev_gra, * dev_val_sum, * dev_gra_min;
		double* dev_hes_val;
		int* dev_pos_x, * dev_pos_y;
		int* dev_csr_index;
		double* temp;
		thrust::pair<int, int>* dev_pos;
		kernel_energy() = default;
		kernel_energy(val_fp _val, gra_fp _gra, hes_fp _hes, valsum_fp _valsum, gramin_fp _gramin, int _size, int _hes_val_size, double* init_x = 0) {
			cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
			val = _val;
			gra = _gra;
			hes = _hes;
			valsum = _valsum;
			gramin = _gramin;
			size = _size;
			hes_val_size = _hes_val_size;
			std::cout << size << hes_val_size << std::endl;
			x = new double[size];

			cudaMalloc((void**)& dev_x, size * sizeof(double));
			cudaMalloc((void**)& dev_gra, size * sizeof(double));
			cudaMalloc((void**)& temp, size * sizeof(double));
			cudaMalloc((void**)& dev_val_sum, sizeof(double));
			cudaMalloc((void**)& dev_gra_min, sizeof(double));
			cudaMalloc((void**)& dev_val, size * sizeof(double));
			cudaMalloc((void**)& dev_pos_x, (hes_val_size + 1) * sizeof(int));
			cudaMalloc((void**)& dev_pos_y, (hes_val_size + 1) * sizeof(int));
			cudaMalloc((void**)& dev_csr_index, (size + 1) * sizeof(int));
			cudaMalloc((void**)& dev_hes_val, hes_val_size * sizeof(double));
			cudaMemcpy(dev_x, init_x, size * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
			cudaMemcpy(x, init_x, size * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToHost);
			cudaMalloc((void**)& dev_pos, hes_val_size * sizeof(thrust::pair<int, int>));
		}
		void calc_val() {
			val << <32 * numSMs, 256 >> > (dev_x, dev_val, size);
			cudaThreadSynchronize();
			valsum << <1, size >> > (dev_val, dev_val_sum, size);
		}

		void calc_gra() {
			gra << <32 * numSMs, 256 >> > (dev_x, dev_gra, size);
			cudaThreadSynchronize();
		}
		double max_abs_gra() {
			double re;
			cudaMemcpy(temp, dev_gra, size * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
			gramin << <1, size >> > (temp, dev_gra_min, size);
			cudaThreadSynchronize();
			cudaMemcpy(&re, dev_gra_min, sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
			re = fabs(re);
			return re;

		}
		void devx_add_vec(double* vec, constuctor_type arrayType) {
			cudaMemcpy(temp, vec, size * sizeof(double), (cudaMemcpyKind)arrayType);
			add_vec << <32 * numSMs, 256 >> > (dev_x, temp, size);
			cudaThreadSynchronize();
		}

		void calc_hes(int hes_cal_index_size = 0) {		//传入生成矩阵的kernel的size大小
			if (hes_cal_index_size == 0) hes_cal_index_size = size;
			hes << <32 * numSMs, 256 >> > (dev_x, dev_pos, dev_hes_val, hes_cal_index_size);
			cudaThreadSynchronize();
			//getchar();
			thrust::device_ptr<double> dev_data_ptr(dev_hes_val);
			thrust::device_ptr<thrust::pair<int, int>> dev_keys_ptr(dev_pos);
			thrust::sort_by_key(dev_keys_ptr, dev_keys_ptr + hes_val_size, dev_data_ptr);
			decouple_pos << <32 * numSMs, 256 >> > (dev_pos, dev_pos_x, dev_pos_y, hes_val_size);
			cudaThreadSynchronize();
			cusparseHandle_t handle;
			cusparseStatus_t status = cusparseCreate(&handle);
			cusparseXcoo2csr(handle, dev_pos_x, hes_val_size, size, dev_csr_index, CUSPARSE_INDEX_BASE_ZERO);

		}

	};

}
#endif