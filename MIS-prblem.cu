
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <cuda_runtime.h>
#include <cstddef>


__global__ void misKernel(vertexId* d_graph, char* d_Flags, bool* d_V, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        size_t v = i;
        while (1) {
            if (d_Flags[v]) break;
            if (atomicCAS(&d_V[v], false, true) == false) {
                size_t k = 0;
                for (size_t j = 0; j < d_graph[v].degree; j++) {
                    vertexId ngh = d_graph[v].Neighbors[j];
                    if (d_Flags[ngh] == 2 || atomicCAS(&d_V[ngh], false, true) == false) {
                        k++;
                    }
                    else break;
                }
                if (k == d_graph[v].degree) {
                    d_Flags[v] = 1;
                    for (size_t j = 0; j < d_graph[v].degree; j++) {
                        vertexId ngh = d_graph[v].Neighbors[j];
                        if (d_Flags[ngh] != 2) d_Flags[ngh] = 2;
                    }
                } else {
                    d_V[v] = false;
                    for (size_t j = 0; j < k; j++) {
                        vertexId ngh = d_graph[v].Neighbors[j];
                        if (d_Flags[ngh] != 2) d_V[ngh] = false;
                    }
                }
            }
        }
    }
}

sequence<char> maximalIndependentSet(Graph const &G) {
    Graph G;
    initializeGraph(G);

    char* Flags = new char[G.n]();
    bool* V = new bool[G.n]();

    char* d_Flags;
    bool* d_V;
    vertexId* d_graph;
    cudaMalloc(&d_Flags, G.n * sizeof(char));
    cudaMalloc(&d_V, G.n * sizeof(bool));

    cudaMemcpy(d_Flags, Flags, G.n * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, G.n * sizeof(bool), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (G.n + blockSize - 1) / blockSize;

    misKernel<<<numBlocks, blockSize>>>(d_graph, d_Flags, d_V, G.n);
    cudaDeviceSynchronize();

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "misKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaMemcpy(Flags, d_Flags, G.n * sizeof(char), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < G.n; ++i) {
        std::cout << "Vertex " << i << ": " << static_cast<int>(Flags[i]) << std::endl;
    }

Error:
    cudaFree(d_Flags);
    cudaFree(d_V);

    delete[] Flags;
    delete[] V;

    return 0;
}

