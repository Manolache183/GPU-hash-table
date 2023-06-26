#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctime>
#include <sstream>
#include <string>
#include "test_map.hpp"
#include "gpu_hashtable.hpp"

using namespace std;

/*
Allocate CUDA memory only through glbGpuAllocator
cudaMalloc -> glbGpuAllocator->_cudaMalloc
cudaMallocManaged -> glbGpuAllocator->_cudaMallocManaged
cudaFree -> glbGpuAllocator->_cudaFree
atomicMax
atomicCAS
atomicExch
*/

__device__ int kernelHashFunction(int key, int capacity) {
	unsigned long int hash = key;
	hash = hash * 11 + 7;
	hash = hash % capacity;
	return hash;
}

__global__ void kernelInsertKey(int *keys, int* values, int numKeys, std::pair<int, int>* hashTable, int capacity, int *size) {
	// Compute the global element index this thread should process
  	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= numKeys) {
		return;
	}

	int key = keys[i];
	int value = values[i];

	if (key <= 0 || value <= 0) {
		return;
	}

	int hash = kernelHashFunction(key, capacity);
	int j = hash;
	int val;

	while (true) {
		val = atomicCAS(&hashTable[j].first, 0, key);

		if (val == 0) { // Empty slot
			val = atomicExch(&hashTable[j].second, value);
			atomicAdd(size, 1);
			break;
		} else if (val == key) { // Key already exists
			val = atomicExch(&hashTable[j].second, value);
			break;
		}

		// Collision, find next empty slot
		j = (j + 1) % capacity;
	}
}

__global__ void kernelGetKey(int *keys, int *values, int numKeys, std::pair<int, int>* hashTable, int capacity) {
	// Compute the global element index this thread should process
  	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= numKeys) {
		return;
	}

	int targetKey = keys[i];

	int hash = kernelHashFunction(targetKey, capacity);

	int j = hash;
	int key;

	while (true) {
		key = hashTable[j].first;

		if (key == 0) { // empty spot
			values[i] = 0;
			return;
		} else if (key == targetKey) { // key found
			values[i] = hashTable[j].second;
			return;
		}

		j = (j + 1) % capacity;
	}
}

__global__ void kernelGetAllPairs(int *keys, int *values, std::pair<int, int>* hashTable, int capacity) {
	// Compute the global element index this thread should process
  	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= capacity) {
		return;
	}

	keys[i] = hashTable[i].first;
	values[i] = hashTable[i].second;
}

float loadFactor(int size, int capacity) {
	return ((float) size / (float) capacity);
}

/**
 * Function constructor GpuHashTable
 * Performs init
 * Example on using wrapper allocators _cudaMalloc and _cudaFree
 */
GpuHashTable::GpuHashTable(int initialCapacity) {
	capacity = initialCapacity;
	glbGpuAllocator->_cudaMalloc((void**) &hashTable, capacity * sizeof(std::pair<int, int>));

	cudaMemset(hashTable, 0, capacity * sizeof(std::pair<int, int>));

	this->size = 0;
}

/**
 * Function desctructor GpuHashTable
 */
GpuHashTable::~GpuHashTable() {
	glbGpuAllocator->_cudaFree(hashTable);
}

/**
 * Function reshape
 * Performs resize of the hashtable based on load factor
 */
void GpuHashTable::reshape(int sizeReshape) {
	std::pair<int, int> *newHashTable;

	glbGpuAllocator->_cudaMalloc((void**) &newHashTable, sizeReshape * sizeof(std::pair<int, int>));

	cudaMemset(newHashTable, 0, sizeReshape * sizeof(std::pair<int, int>));

	int *device_keys, *device_values, *device_size;
	this->size = 0;

	glbGpuAllocator->_cudaMalloc((void**) &device_keys, this->capacity * sizeof(int));
	glbGpuAllocator->_cudaMalloc((void**) &device_values, this->capacity * sizeof(int));
	glbGpuAllocator->_cudaMalloc((void**) &device_size, sizeof(int));

	cudaMemcpy(device_size, &(this->size), sizeof(int), cudaMemcpyHostToDevice);

	const int blockSize = 1024;
	size_t block_no = this->capacity / blockSize;
	if (this->capacity % blockSize != 0) {
		block_no++;
	}

	kernelGetAllPairs<<<block_no, blockSize>>>(device_keys, device_values, hashTable, this->capacity);
	cudaDeviceSynchronize();

	kernelInsertKey<<<block_no, blockSize>>>(device_keys, device_values, this->capacity, newHashTable, sizeReshape, device_size);
	cudaDeviceSynchronize();

	cudaMemcpy(&(this->size), device_size, sizeof(int), cudaMemcpyDeviceToHost);

	glbGpuAllocator->_cudaFree(hashTable);

	glbGpuAllocator->_cudaFree(device_keys);
	glbGpuAllocator->_cudaFree(device_values);
	glbGpuAllocator->_cudaFree(device_size);

	hashTable = newHashTable;
	this->capacity = sizeReshape;
}

/**
 * Function insertBatch
 * Inserts a batch of key:value, using GPU and wrapper allocators
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	if (keys == NULL || values == NULL || numKeys < 0) {
		return false;
	}

	int newSize = this->size + numKeys;

	if (loadFactor(newSize, this->capacity) > 0.8) {
		reshape(3 * newSize / 2);
	}

	int *device_keys, *device_values, *device_size;

	glbGpuAllocator->_cudaMalloc((void**) &device_keys, numKeys * sizeof(int));
	glbGpuAllocator->_cudaMalloc((void**) &device_values, numKeys * sizeof(int));
	glbGpuAllocator->_cudaMalloc((void**) &device_size, sizeof(int));

	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_size, &(this->size), sizeof(int), cudaMemcpyHostToDevice);

	const int blockSize = 1024;
	size_t block_no = numKeys / blockSize;
	if (numKeys % blockSize != 0) {
		block_no++;
	}

	kernelInsertKey<<<block_no, blockSize>>>(device_keys, device_values, numKeys, hashTable, this->capacity, device_size);
	cudaDeviceSynchronize();

	cudaMemcpy(&(this->size), device_size, sizeof(int), cudaMemcpyDeviceToHost);

	glbGpuAllocator->_cudaFree(device_keys);
	glbGpuAllocator->_cudaFree(device_values);
	glbGpuAllocator->_cudaFree(device_size);

	return true;
}

/**
 * Function getBatch
 * Gets a batch of key:value, using GPU
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	if (keys == NULL || numKeys < 0) {
		return NULL;
	}

	int *values, *device_keys, *device_values;

	values = (int*) malloc(numKeys * sizeof(int));
	if (values == NULL) {
		cout << "Error allocating memory for values" << endl;
		exit(1);
	}

	glbGpuAllocator->_cudaMalloc((void**) &device_keys, numKeys * sizeof(int));
	glbGpuAllocator->_cudaMalloc((void**) &device_values, numKeys * sizeof(int));

	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	const int blockSize = 1024;
	size_t block_no = numKeys / blockSize;
	if (numKeys % blockSize != 0) {
		block_no++;
	}

	kernelGetKey<<<block_no, blockSize>>>(device_keys, device_values, numKeys, hashTable, capacity);
	cudaDeviceSynchronize();

	cudaMemcpy(values, device_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);


	glbGpuAllocator->_cudaFree(device_keys);
	glbGpuAllocator->_cudaFree(device_values);

	return values;
}