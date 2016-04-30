#include <string>
#include <mkl_types.h>

int readSymmSparseMatrix(const std::string& filename,
                         MKL_INT &n,
                         MKL_INT *&ia,
                         MKL_INT *&ja,
                         double *&va);
