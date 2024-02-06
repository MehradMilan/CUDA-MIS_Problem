#include <cstdio>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "common/graph.h"
#include "MIS.h"
#include <ctime>
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <curand_kernel.h>
#include <ctime>


const int blockSize = 1024;
const int N = 8192;
const int ceilN = (N + blockSize - 1) / blockSize;
const int ceilNN = (N * N + blockSize - 1) / blockSize;

inline void checkError(cudaError_t error, int line) {
	if (error != cudaSuccess) {
		printf("CUDA error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, line);
		exit(EXIT_FAILURE);
	}
}

inline void checkCurandError(curandStatus_t status, int line) {
	if (status != CURAND_STATUS_SUCCESS) {
		printf("CURAND error(code %d), line(%d)\n", status, line);
		exit(EXIT_FAILURE);
	}
}

__global__ void setOnes(int* Array) {
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	Array[thid] = 1;
}

__global__ void indFind(int N, int* SubGraph, int* Adj, unsigned int* Random) {

	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	
	int u = thid / N;
	int v = thid % N;

	if (v >= u) return;
	
	bool bit = Random[u ^ v] & 1;
	if (Adj[thid] && SubGraph[u] && SubGraph[v]) {
		SubGraph[u * bit + v * (1 - bit)] = 0;
	}
}

__global__ void updateWithInd(int N, int* MIS, int* Graph, int* IndSet) {
	int thid = blockIdx.x * blockDim.x + threadIdx.x;

	bool indSetThid = IndSet[thid];

	MIS[thid] |= indSetThid;

	Graph[thid] &= !indSetThid;
}

__global__ void updateWithNeighs(int N, int* Graph, int* Adj, int* IndSet) {
	int thid = blockIdx.x * blockDim.x + threadIdx.x;

	int u = thid / N;
	int v = thid % N;

	if (v >= u) return;

	if (Adj[thid]) {
		if (IndSet[u]) Graph[v] = 0;
		else if (IndSet[v]) Graph[u] = 0;
	}
}

__global__ void blockReduction(int* ArrayGlob, int* BlockSums) {
	__shared__ int shData[1024];
	__shared__ int* Array;

	int thid = threadIdx.x;
	if (thid == 0) Array = ArrayGlob + blockDim.x * blockIdx.x;
	__syncthreads();

	shData[thid] = Array[thid];
	__syncthreads();

	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
		if (thid < offset) shData[thid] += shData[thid + offset];
		__syncthreads();
	}

	if (thid == 0) BlockSums[blockIdx.x] = shData[0];
}

__global__ void correctEdges(int N, int* SubGraph, int* NewAdj, int* Adj) {
	int thid = blockIdx.x * blockDim.x + threadIdx.x;

	int u = thid / N;
	int v = thid % N;

	NewAdj[thid] = (Adj[thid] && SubGraph[u] && SubGraph[v]);
}

__global__ void markHeavy(int* Degrees, int* HeavySet, int lowerbound) {
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	HeavySet[thid] = Degrees[thid] >= lowerbound;
}

__global__ void removeVertices(int* WithHeavySubset, int* HeavySet) {
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	WithHeavySubset[thid] &= !HeavySet[thid];
}

__global__ void checkEdgesAndMarkNeighs(int N, int* MIS, int* Adj, int* Marked, int* flag) {
	int thid = blockIdx.x * blockDim.x + threadIdx.x;

	int u = thid / N;
	int v = thid % N;

	bool chosenU = MIS[u];
	bool chosenV = MIS[v];

	if (Adj[thid]) {
		if (chosenU && chosenV) *flag = 1;
		else if (chosenU || chosenV) Marked[u] = Marked[v] = 1;
	}
	if (u == v && MIS[u])
		Marked[u] = 1;
}

__global__ void checkIfMarked(int* Marked, int* flag) {
	int thid = blockIdx.x * blockDim.x + threadIdx.x;

	if (!Marked[thid]) *flag = 1;
}

__global__ void randChoice(int* Subset, unsigned int* Choice, int* HeavySet) {
	int thid = blockIdx.x * blockDim.x + threadIdx.x;

	if (HeavySet[thid] && (Choice[thid] % 32 == 0))
		Subset[thid] = 1;
}

__global__ void prepareCountingScore(int N, int* ScoreArray, int* Adj, int* CurrentSet, int* WithHeavySubset) {
	int thid = blockIdx.x * blockDim.x + threadIdx.x;

	int u = thid / N;
	int v = thid % N;
	
	if (u > v) return;

	bool chosenU = CurrentSet[u];
	bool chosenV = CurrentSet[v];

	if (u == v && chosenU) {
		ScoreArray[thid] = 1;
		return;
	}

	if (chosenU && chosenV) {
		ScoreArray[thid] = -1;
	}

	if (chosenU || chosenV) {
		ScoreArray[thid] = 1;
	}
}


void showoff() {
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
			prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
			2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
	}
}


void initializeHost(int N, int *& MIS, int *& Adj) {
	MIS = new int[N]();
	Adj = new int[N*N]();
}

void initializeDevice(int N, int*& MIS, int*& Adj, int*& MISDev, int*& AdjDev, int*&CurrentGraph) {
	checkError(cudaMalloc((void**)&MISDev, N * sizeof(int)), __LINE__);
	checkError(cudaMalloc((void**)&AdjDev, N * N * sizeof(int)), __LINE__);
	checkError(cudaMalloc((void**)&CurrentGraph, N * sizeof(int)), __LINE__);
	checkError(cudaMemcpy(AdjDev, Adj, N * N * sizeof(int), cudaMemcpyHostToDevice), __LINE__);
	checkError(cudaMemset(MISDev, 0, N * sizeof(int)), __LINE__);
}

void freeMemory(int*& MISDev, int*& AdjDev, int*& CurrentGraphDev, int*& MIS, int*& Adj) {
	cudaFree(MISDev);
	cudaFree(AdjDev);
	cudaFree(CurrentGraphDev);
	delete[] MIS;
	delete[] Adj;
}

void initializeDeviceSupport(int N, int ceilN, int*& WithHeavySubset, int*& HeavySet, int*& ScoreSet, int*& Degrees, unsigned int*& RandomChoice) {
	checkError(cudaMalloc((void**)&WithHeavySubset, N * sizeof(int)), __LINE__);
	checkError(cudaMalloc((void**)&HeavySet, N * sizeof(int)), __LINE__);
	checkError(cudaMalloc((void**)&ScoreSet, N * sizeof(int)), __LINE__);
	checkError(cudaMalloc((void**)&Degrees, N * sizeof(int)), __LINE__);
	checkError(cudaMalloc((void**)&RandomChoice, ceilN * sizeof(int)), __LINE__);
}

void freeDeviceSupport(int*& WithHeavySubset, int*& HeavySet, int*& ScoreSet, int*& Degrees, unsigned int*& RandomChoice) {
	cudaFree(WithHeavySubset);
	cudaFree(HeavySet);
	cudaFree(ScoreSet);
	cudaFree(Degrees);
	cudaFree(RandomChoice);
}


void solve(int*& MIS, int*& MISDev, int *&AdjDev, int *&CurrentGraphDev);
void checkMIS(int*& MIS, int*& Adj);

parlay::sequence<char> maximalIndependentSet(Graph const &G) {
	showoff();

	int *MIS, *Adj;
	initializeHost(G.n, MIS, Adj);

	int t;
	for (size_t i = 0; i < G.n; i++) {
		for (size_t j = 0; j< G[i].degree; j++) {
			t = G[i].Neighbors[j];
			Adj[i * G.n + t] = 1;
		}
    }

	int* MISDev, *AdjDev, *CurrentGraph;
	initializeDevice(N, MIS, Adj, MISDev, AdjDev, CurrentGraph);

	solve(MIS, MISDev, AdjDev, CurrentGraph);

	parlay::sequence<char> Flags(G.n, (char) 0);
	for (size_t i=0; i< G.n; i++){
		if ((int) MIS[i])
			Flags[i] = 2;
		else
			Flags[i] = 1;
	}

	freeMemory(MISDev, AdjDev, CurrentGraph, MIS, Adj);

	
	return Flags;
}


int countLeft(int*& CurrentGraph);
void heavyFind(int currentCardinality, int*& CurrentGraph, int*& Adj, int*& Degrees, int*& WithHeavySubset, int*& HeavySet, int& heavySetCardinality);
void scoreFind(int*& Adj, int*& WithHeavySubset, int*& HeavySet, int heavySetCardinality, int*& ScoreFind, unsigned int*& RandomChoice, curandGenerator_t& rand_gen);

void solve(int*& MIS, int *& MISDev, int *&Adj, int *&CurrentGraph) {
	setOnes<<<ceilN, blockSize>>>(CurrentGraph);
	
	unsigned int* RandomChoice;
	curandGenerator_t rand_gen;
	checkCurandError(curandCreateGenerator(&rand_gen, CURAND_RNG_PSEUDO_DEFAULT), __LINE__);
	checkCurandError(curandSetPseudoRandomGeneratorSeed(rand_gen, 1234ULL), __LINE__);

	int* WithHeavySubset;
	int* HeavySet;
	int* ScoreSet;
	int* IndSet;
	int* Degrees;

	initializeDeviceSupport(N, ceilN, WithHeavySubset, HeavySet, ScoreSet, Degrees, RandomChoice);

	int currentCardinality = N;
	while (currentCardinality) {
		checkError(cudaMemset(HeavySet, 0, N * sizeof(int)), __LINE__);
		checkError(cudaMemset(ScoreSet, 0, N * sizeof(int)), __LINE__);
		checkError(cudaMemset(Degrees, 0, N * sizeof(int)), __LINE__);

		int heavySetCardinality;
		heavyFind(currentCardinality, CurrentGraph, Adj, Degrees, WithHeavySubset, HeavySet, heavySetCardinality);		//O(log^2 N)

		scoreFind(Adj, WithHeavySubset, HeavySet, heavySetCardinality, ScoreSet, RandomChoice, rand_gen);		//O(log N)
		
		checkCurandError(curandGenerate(rand_gen, RandomChoice, N), __LINE__);
		IndSet = ScoreSet;
		indFind << < ceilNN, blockSize >> > (N, IndSet, Adj, RandomChoice);
		updateWithInd << <ceilN, blockSize >> > (N, MISDev, CurrentGraph, IndSet);
		updateWithNeighs << <ceilNN, blockSize >> > (N, CurrentGraph, Adj, IndSet);
		cudaDeviceSynchronize();

		int newCardinality = countLeft(CurrentGraph);
		if (newCardinality == currentCardinality) break;
		currentCardinality = newCardinality;
	}

	int* BrutalChosen = ScoreSet;
	while (currentCardinality) {
		checkError(cudaMemcpy(BrutalChosen, CurrentGraph, N * sizeof(int), cudaMemcpyDeviceToDevice), __LINE__);
		checkCurandError(curandGenerate(rand_gen, RandomChoice, N), __LINE__);
		indFind << < ceilNN, blockSize >> > (N, BrutalChosen, Adj, RandomChoice);
		updateWithInd << <ceilN, blockSize >> > (N, MISDev, CurrentGraph, BrutalChosen);
		updateWithNeighs << <ceilNN, blockSize >> > (N, CurrentGraph, Adj, BrutalChosen);
		cudaDeviceSynchronize();
		currentCardinality = countLeft(CurrentGraph);
	}

	checkError(cudaMemcpy(MIS, MISDev, N * sizeof(int), cudaMemcpyDeviceToHost), __LINE__);

	freeDeviceSupport(WithHeavySubset, HeavySet, ScoreSet, Degrees, RandomChoice);

}


void heavyFind(int currentCardinality, int*& CurrentGraph, int*& Adj, int*& Degrees, int*& WithHeavySubset, int*& HeavySet, int& heavySetCardinality) {
	int target = (int)(currentCardinality / log2(currentCardinality));


	checkError(cudaMemcpy(WithHeavySubset, CurrentGraph, N * sizeof(int), cudaMemcpyDeviceToDevice), __LINE__);

	int* CurrentAdj;
	checkError(cudaMalloc((void**)&CurrentAdj, N * N * sizeof(int)), __LINE__);

	int* BlockSumsHost = new int[ceilNN];
	int* BlockSumsDev;
	checkError(cudaMalloc((void**)&BlockSumsDev, ceilNN * sizeof(int)), __LINE__);

	int step = (int)ceil(log2(currentCardinality));
	while (step) {
		step--;
		checkError(cudaMemset(HeavySet, 0, N * sizeof(int)), __LINE__);

		int* hewi = new int[N]();
		cudaMemcpy(hewi, WithHeavySubset, N * sizeof(int), cudaMemcpyDeviceToHost);

		correctEdges << <ceilNN, blockSize >> > (N, WithHeavySubset, CurrentAdj, Adj);

		blockReduction << < ceilNN, blockSize >> > (CurrentAdj, BlockSumsDev);
		checkError(cudaMemcpy(BlockSumsHost, BlockSumsDev, ceilNN * sizeof(int), cudaMemcpyDeviceToHost), __LINE__);

		int* DegreesHost = new int[N]();

		for (int i = 0; i < ceilNN; i++)
			DegreesHost[i / ceilN] += BlockSumsHost[i];

		checkError(cudaMemcpy(Degrees, DegreesHost, N * sizeof(int), cudaMemcpyHostToDevice), __LINE__);
		delete[] DegreesHost;

		markHeavy << <ceilN, blockSize >> > (Degrees, HeavySet, std::max(1,(1 << step) - 1));
		blockReduction << <ceilN, blockSize >> > (HeavySet, BlockSumsDev);
		checkError(cudaMemcpy(BlockSumsHost, BlockSumsDev, ceilN * sizeof(int), cudaMemcpyDeviceToHost), __LINE__);

		heavySetCardinality = 0;
		for (int i = 0; i < ceilN; i++)
			heavySetCardinality += BlockSumsHost[i];
		
		if (heavySetCardinality >= target) break;
		else {
			removeVertices << <ceilN, blockSize >> > (WithHeavySubset, HeavySet);
			cudaDeviceSynchronize();
		}
	}
	cudaFree(CurrentAdj);
	cudaFree(BlockSumsDev);
	delete[] BlockSumsHost;
}

void scoreFind(int*& Adj, int*& WithHeavySubset, int*& HeavySet, int heavySetCardinality, int*& ScoreSet, unsigned int*& RandomChoice, curandGenerator_t& rand_gen) {
	int cardinality = std::max(2, heavySetCardinality >> 5);

	int bestScore = -INT_MAX;
	int* BestSet = new int[N]();

	int currentScore;
	int* CurrentSet;
	int* CurrentSetHost = new int[N];
	checkError(cudaMalloc((void**)&CurrentSet, N * sizeof(int)), __LINE__);

	int* ScoreArray;
	checkError(cudaMalloc((void**)&ScoreArray, N * N * sizeof(int)), __LINE__);

	int* BlockSumsHost = new int[ceilNN];
	int* BlockSumsDev;
	checkError(cudaMalloc((void**)&BlockSumsDev, ceilNN * sizeof(int)), __LINE__);

	for (int i = 0; i < 10; i++) {
		checkError(cudaMemset(CurrentSet, 0, N * sizeof(int)), __LINE__);
		checkError(cudaMemset(ScoreArray, 0, N * N * sizeof(int)), __LINE__);
		checkError(cudaMemset(BlockSumsDev, 0, ceilNN * sizeof(int)), __LINE__);
		currentScore = 0;

		checkCurandError(curandGenerate(rand_gen, RandomChoice, N), __LINE__);

		randChoice << <ceilN, blockSize >> > (CurrentSet, RandomChoice, HeavySet);
		checkError(cudaMemcpy(CurrentSetHost, CurrentSet, N * sizeof(int), cudaMemcpyDeviceToHost), __LINE__);

		prepareCountingScore << <ceilNN, blockSize >> > (N, ScoreArray, Adj, CurrentSet, WithHeavySubset); //O(1)
		blockReduction << <ceilNN, blockSize >> > (ScoreArray, BlockSumsDev);	//O(1)
		checkError(cudaMemcpy(BlockSumsHost, BlockSumsDev, ceilNN * sizeof(int), cudaMemcpyDeviceToHost), __LINE__);

		for (int i = 0; i < ceilNN; i++)
			if (!CurrentSetHost[i / blockSize] && BlockSumsHost[i]) BlockSumsHost[i] = 1;

		for (int i = 0; i < ceilNN; i++)
			currentScore += BlockSumsHost[i];

		if (currentScore > bestScore) {
			bestScore = currentScore;
			checkError(cudaMemcpy(BestSet, CurrentSet, N * sizeof(int), cudaMemcpyDeviceToHost), __LINE__);
		}
	}

	checkError(cudaMemcpy(ScoreSet, BestSet, N * sizeof(int), cudaMemcpyHostToDevice), __LINE__);

	delete[] BestSet;
	delete[] BlockSumsHost;
	delete[] CurrentSetHost;
	cudaFree(CurrentSet);
	cudaFree(ScoreArray);
	cudaFree(BlockSumsDev);
}


int countLeft(int*& CurrentGraph) {
	int result = 0;
	int* BlockSumsDev;
	checkError(cudaMalloc((void**)&BlockSumsDev, ceilN * sizeof(int)), __LINE__);
	int* BlockSumsHost = new int[ceilN];

	blockReduction << <ceilN, blockSize >> > (CurrentGraph, BlockSumsDev);
	checkError(cudaMemcpy(BlockSumsHost, BlockSumsDev, ceilN * sizeof(int), cudaMemcpyDeviceToHost), __LINE__);

#pragma omp parallel for schedule(static) reduction(+:result)
	for (int i = 0; i < ceilN; i++)
		result += BlockSumsHost[i];

	cudaFree(BlockSumsDev);
	delete[] BlockSumsHost;

	return result;
}

