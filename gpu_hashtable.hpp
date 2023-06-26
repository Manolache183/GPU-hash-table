#ifndef _HASHCPU_
#define _HASHCPU_

/**
 * Class GpuHashTable to implement functions
 */
class GpuHashTable
{
	public:
		GpuHashTable(int capacity);
		void reshape(int sizeReshape);

		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);

		~GpuHashTable();

	private:
		std::pair<int, int> *hashTable;
		int size;
		int capacity;
};

#endif
