#include <iostream>
#include <cstdlib>
#include <ctime>

int* generateRandomGraph(int N) {
    srand(time(NULL));

    int* G = new int[N * N]();

    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            int edge = rand() % 2;
            G[i * N + j] = edge;
            G[j * N + i] = edge;
        }
    }

    return G;
}