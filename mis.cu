#include <curand_kernel.h>

#define BLOCK_SIZE 1024

__global__ void initArray(int* Array, int value) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	Array[tid] = value;
}

void initializeDevice(int*& G, int*& G_dev, int*& I, int*& I_dev, int*&H, int n) {
	cudaMalloc((void**)&I_dev, n * sizeof(int));
	cudaMalloc((void**)&G_dev, n * n * sizeof(int));
	cudaMalloc((void**)&H, n * sizeof(int));

	initArray<<<n, BLOCK_SIZE>>>(H, 1);

	cudaMemcpy(G_dev, G, n * n * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemset(I_dev, 0, n * sizeof(int));
}

bool findI(int *G, int *I, int n) {
	// G: Adj
	// I: MIS
	// n: N
	// G_dev: AdjDev
	// I_dev: MISDev
	// H: currentGraph
	// K: WithHeavySubset
	// M: HeavySet
	// T: ScoreSet
	// S: IndSet
	// D: Degrees

	int *G_dev, *I_dev, *H;
    initializeDevice(G, G_dev, I, I_dev, H, n);
	int *K, *M, *T, *S, *D;

	unsigned int* RandomChoice;
	curandGenerator_t rand_gen;
	curandCreateGenerator(&rand_gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(rand_gen, 1234ULL);

	int currentGSize = n;
	while(currentGSize > 0) {
		cudaMemset(M, 0, n * sizeof(int));
		cudaMemset(T, 0, n * sizeof(int));
		cudaMemset(D, 0, n * sizeof(int));

		int currentHSize = findM(G, H, D, M, currentGSize);
		
		scoreFind(G, K, M, currentHSize, T, RandomChoice, rand_gen);		//O(log N)
		
		curandGenerate(rand_gen, RandomChoice, n);
		S = T;
		// indFind << < ceilNN, blockSize >> > (N, IndSet, Adj, RandomChoice);		//O(1)
		// updateWithInd << <ceilN, blockSize >> > (N, MISDev, CurrentGraph, IndSet);	//O(1)
		// updateWithNeighs << <ceilNN, blockSize >> > (N, CurrentGraph, Adj, IndSet);	

		// int newCardinality = countLeft(CurrentGraph);
		// if (newCardinality == currentCardinality) break;
		// currentCardinality = newCardinality;

	}

	int* BrutalChosen = T;	//steal memory
	while (currentGSize) {
	// 	checkError(cudaMemcpy(BrutalChosen, CurrentGraph, N * sizeof(int), cudaMemcpyDeviceToDevice), __LINE__);
	// 	checkCurandError(curandGenerate(rand_gen, RandomChoice, N), __LINE__);
	// 	indFind << < ceilNN, blockSize >> > (N, BrutalChosen, Adj, RandomChoice);	//O(1)
	// 	updateWithInd << <ceilN, blockSize >> > (N, MISDev, CurrentGraph, BrutalChosen);	//O(1)
	// 	updateWithNeighs << <ceilNN, blockSize >> > (N, CurrentGraph, Adj, BrutalChosen);	//O(1)
	// 	cudaDeviceSynchronize();
	// 	currentCardinality = countLeft(CurrentGraph);
	}

	cudaMemcpy(I, I_dev, n * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(G_dev);
	cudaFree(I_dev);
	cudaFree(H);
	cudaFree(K);
	cudaFree(M);
	cudaFree(T);
	cudaFree(S);
	cudaFree(D);
	cudaFree(RandomChoice);

	return true;
}

int findM(int *&G, int *&H, int *&D, int *&M, int currentGSize) {

}

void scoreFind(int*& Adj, int*& WithHeavySubset, int*& HeavySet, int heavySetCardinality, int*& ScoreSet, unsigned int*& RandomChoice, curandGenerator_t& rand_gen) {}