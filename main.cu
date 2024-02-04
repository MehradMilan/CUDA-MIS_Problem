#include "stdio.h"
#include "utility.cpp"
#include "mis.cu"

int main() {
    int n;
    scanf("%d", &n);
    int *I = new int[n]();
    int *G = generateRandomGraph(n);
    bool success = findI(G, I, n);

    delete[] G;
    delete[] I;
    return 0;
}