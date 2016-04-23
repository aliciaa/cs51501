#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"

int readSymmSparseMatrix(char* filename, int& n, int& nz, int*& i, int*& j, double*& v);

int readSymmMatrix(char* filename, int& n, double*& v);

int readMatrix(char* filename, int& m, int& n, double*& v, bool is_vector = false);

int readBandedMatrix(char* filename, int& n, int& t, int& nz, long long *& AR, long long *& AC, double *& AV);

int writeMatrix(char* filename, int m, int n, double* v);
