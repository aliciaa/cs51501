#include "matio.h"
#include "mmio.h"
#include <cstdio>

int readSymmSparseMatrix(const std::string &filename,
                         MKL_INT &n,
                         MKL_INT *&ia,
                         MKL_INT *&ja,
                         double *&va)
{
	int in, im, nz;

	// try opening the files
	FILE *fp;
	if ((fp = fopen(filename.c_str(), "r")) == NULL) {
		fprintf(stderr, "Error occurs while reading from file %s.\n", filename.c_str());
		return 1;
	}

	// try reading the banner
	MM_typecode type;
	if (mm_read_banner(fp, &type) != 0) {
		fprintf(stderr, "Could not process Matrix Market banner.\n");
		return 2;
	}

	// check the type
	if (!mm_is_matrix(type) || !mm_is_coordinate(type) || !mm_is_real(type) || !mm_is_symmetric(type)) {
		fprintf(stderr, "Sorry, this application does not support Market Market type: [%s]\n",
				mm_typecode_to_str(type));
		return 3;
	}

	// read the sizes of the vectors
	if (mm_read_mtx_crd_size(fp, &in, &im, &nz)) {
		fprintf(stderr, "Could not read the size of the matrix.\n");
		return 4;
	}

	n = in;

	// check if it is a square matrix
	if (in != im) {
		fprintf(stderr, "Needs to be square.\n");
		return 5;
	}

	// allocate the memory
	printf("reading %s:\n\ta %d x %d sparse matrix with %d nonzeros...", filename.c_str(), in, im, nz);
	int *cooi = new int[nz];
	int *cooj = new int[nz];
	double *coov = new double[nz];
	ia = new MKL_INT[n+1]();
  int nnz = 0;
	for (int k = 0; k < nz; ++k) {
		fscanf(fp, "%u %u %lf\n", &cooj[k], &cooi[k], &coov[k]);
		++ia[cooi[k]];
    ++nnz;
		if (cooi[k] != cooj[k]) {
			++ia[cooj[k]];
      ++nnz;
		}
	}
	printf("done\n");

	// close the file
	fclose(fp);

  ia[0] = 1;
  for (int k = 1; k <= n; ++k) {
    ia[k] += ia[k-1];
  }

  // convert it to CSR format full matrix
  ja = new MKL_INT[nnz];
  va = new double[nnz];
  MKL_INT *pt = new MKL_INT[n];
  for (int k = 0; k < n; ++k) {
    pt[k] = ia[k] - 1;
  }
  for (int k = 0; k < nz; ++k) {
    ja[pt[cooi[k]-1]] = cooj[k];
    va[pt[cooi[k]-1]] = coov[k];
    ++pt[cooi[k]-1];
    if (cooi[k] != cooj[k]) {
      ja[pt[cooj[k]-1]] = cooi[k];
      va[pt[cooj[k]-1]] = coov[k];
      ++pt[cooj[k]-1];
    }
  }

  delete [] cooi;
  delete [] cooj;
  delete [] coov;
  delete [] pt;

	return 0;
}
