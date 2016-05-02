#include "matio.h"
#include <algorithm>
#include <math.h>

int readSparseMatrix(char * filename, PetscInt & n, PetscInt & nz, PetscInt *& nnz, int *& i, int *& j, double *& v)
{
	int in, m, inz;

	// try opening the files
	FILE *fp;
	if ((fp = fopen(filename, "r")) == NULL) {
		fprintf(stderr, "Error occurs while reading from file %s.\n", filename);
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
	if (mm_read_mtx_crd_size(fp, &in, &m, &inz)) {
		fprintf(stderr, "Could not read the size of the matrix.\n");
		return 4;
	}

	n = in;
	nz = inz;

	// check if it is a square matrix
	if (in != m) {
		fprintf(stderr, "Needs to be square.\n");
		return 5;
	}

	// allocate the memory
	printf("reading %s:\n\ta %d x %d sparse matrix with %d nonzeros...", filename, m, in, inz);
	i = new int[nz];
	j = new int[nz];
	v = new double[nz];
	nnz = new PetscInt[n]();
	for (int k = 0; k < inz; ++k) {
		fscanf(fp, "%u %u %lf\n", &j[k], &i[k], &v[k]);
		--i[k];
		--j[k];
		++nnz[i[k]];
		if (i[k] != j[k]) {
			++nnz[j[k]];
		}
	}
	printf("done\n");

	// close the file
	fclose(fp);

	return 0;
}

int readSymmMatrix(char* filename, int& n, double*& v)
{
	int m;

	// try opening the files
	FILE *fp;
	if ((fp = fopen(filename, "r")) == NULL) {
		fprintf(stderr, "Error occurs while reading from file %s.\n", filename);
		return 1;
	}

	// try reading the banner
	MM_typecode type;
	if (mm_read_banner(fp, &type) != 0) {
		fprintf(stderr, "Could not process Matrix Market banner.\n");
		return 2;
	}

	// check the type
	if (!mm_is_matrix(type) || !mm_is_array(type) || !mm_is_real(type) || !mm_is_symmetric(type)) {
		fprintf(stderr, "Sorry, this application does not support Market Market type: [%s]\n",
				mm_typecode_to_str(type));
		return 3;
	}

	// read the sizes of the vectors
	if (mm_read_mtx_array_size(fp, &m, &n)) {
		fprintf(stderr, "Could not read the size of the matrix.\n");
		return 4;
	}

	// check if it is a square matrix
	if (n != m) {
		fprintf(stderr, "Needs to be square.\n");
		return 5;
	}

	// allocate the memory
	printf("reading %s:\n\ta %d x %d matrix...", filename, m, n);
	v = new double[m * n];
	for (int j = 0; j < n; ++j) {
		for (int i = j; i < m; ++i) {
			fscanf(fp, "%lf\n", &v[i * n + j]);
			if (i != j) {
				v[j * n + i] = v[i * n + j];
			}
		}
	}
	printf("done\n");

	// close the file
	fclose(fp);

	return 0;
}

int readMatrix(char* filename, int& m, int& n, double*& v, bool is_vector)
{
	// try opening the files
	FILE *fp;
	if ((fp = fopen(filename, "r")) == NULL) {
		fprintf(stderr, "Error occurs while reading from file %s.\n", filename);
		return 1;
	}

	// try reading the banner
	MM_typecode type;
	if (mm_read_banner(fp, &type) != 0) {
		fprintf(stderr, "Could not process Matrix Market banner.\n");
		return 2;
	}

	// check the type
	if (!mm_is_matrix(type) || !mm_is_array(type) || !mm_is_real(type) || !mm_is_general(type)) {
		fprintf(stderr, "Sorry, this application does not support Market Market type: [%s]\n",
				mm_typecode_to_str(type));
		return 3;
	}

	// read the sizes of the vectors
	if (mm_read_mtx_array_size(fp, &m, &n)) {
		fprintf(stderr, "Could not read the size of the matrix.\n");
		return 4;
	}

	// check if it is a vector
	if (is_vector && n != 1) {
		fprintf(stderr, "Needs to be a vector.\n");
		return 5;
	}

	// allocate the memory
	printf("reading %s:\n\ta %d x %d matrix...", filename, m, n);
	v = new double[m * n];
	for (int j = 0; j < n; ++j) {
		for (int i = 0; i < m; ++i) {
			fscanf(fp, "%lf\n", &v[j * n + i]);
		}
	}
	printf("done\n");

	// close the file
	fclose(fp);

	return 0;
}

int readBandedMatrix(char* filename, int& n, int& t, int& nz, long long*& AR, long long*& AC, double*& AV)
{
	// try opening the files
	FILE *fp;
	if ((fp = fopen(filename, "r")) == NULL) {
		fprintf(stderr, "Error occurs while reading from file %s.\n", filename);
		return 1;
	}

	// try reading the banner
	MM_typecode type;
	if (mm_read_banner(fp, &type)) {
		fprintf(stderr, "Could not process Matrix Market banner.\n");
		return 2;
	}

	// check the type
	if (!mm_is_matrix(type) || !mm_is_coordinate(type) || !mm_is_real(type) || !mm_is_general(type)) {
		fprintf(stderr, "Sorry, this application does not support Market Market type: [%s]\n",
				mm_typecode_to_str(type));
		return 3;
	}

	// read the sizes and nnz of the matrix
	int m;
	if (mm_read_mtx_crd_size(fp, &n, &m, &nz)) {
		fprintf(stderr, "Could not read the size of the matrix.\n");
		return 4;
	}
	
	printf("reading %s:\n\ta %d x %d banded matrix ", filename, n, m);
	// allocate the memory
	AR = new long long[nz];
	AC = new long long[nz];
	AV = new double[nz];
	t = 0;
	for (int i = 0; i < nz; ++i) {
		fscanf(fp, "%d %d %lf\n", AR + i, AC + i, AV + i);
		--AR[i];		// 0-indexing
		--AC[i];		// 0-indexing
		t = std::max(t, (int)(AR[i] - AC[i]));
	}
	printf("with bandwidth (2m + 1) = %d...done\n", 2 * t + 1);

	// close the file
	fclose(fp);

	return 0;
}

int writeMatrix(char* filename, int m, int n, double* v)
{
	// try opening the file
	FILE *fp;
	if ((fp = fopen(filename, "w")) == NULL) {
		fprintf(stderr, "Error occurs while writing to file %s\n", filename);
		return 1;
	}

	// set the type
	MM_typecode type;
	mm_initialize_typecode(&type);
	mm_set_matrix(&type);
	mm_set_array(&type);
	mm_set_real(&type);
	mm_set_general(&type);

	// write banner and sizes
	mm_write_banner(fp, type);
	mm_write_mtx_array_size(fp, m, n);

	printf("writing %s:\n\ta %d x %d matrix...", filename, m, n);
	for (int j = 0; j < n; ++j) {
		for (int i = 0; i < m; ++i) { 
			fprintf(fp, "%lf\n", v[i * n + j]);
		}
	}
	printf("done\n");

	// close the file
	fclose(fp);

	return 0;
}

int MatRead(char *fileA, PetscInt& n, Mat& A)
{
	PetscInt *arrayIA, *arrayJA,			// arrays of rows and columns of A
					 *nnzA,										// number of nonzeros in each row
					 nzA;											// number of nonzeros in matrix A
	PetscScalar *arrayVA;							// array of values of A
	int read_error = 0;

	if (readSparseMatrix(fileA, n, nzA, nnzA, arrayIA, arrayJA, arrayVA)) {
		// read_error in reading in the vector, set read_error to 1
		read_error = 1;
	}

	/*---------------------------------------------------------------------------
	 * set up the matrix A
	 *---------------------------------------------------------------------------*/
	MatCreateSeqAIJ(MPI_COMM_SELF, n, n, nzA, nnzA, &A);

	/*---------------------------------------------------------------------------
	 * set the values of matrix A
	 *---------------------------------------------------------------------------*/
	/*
		 PetscMalloc1(n * sizeof(PetscInt), &idx);
		 for (PetscInt i = 0; i < n; ++i) {
		 idx[i] = i;
		 }
		 MatSetValues(A, n, idx, n, idx, arrayA, INSERT_VALUES);
		 */
	for (PetscInt i = 0; i < nzA; ++i) {
		MatSetValue(A, arrayIA[i], arrayJA[i], arrayVA[i], INSERT_VALUES);
		if (arrayIA[i] != arrayJA[i]) {
			MatSetValue(A, arrayJA[i], arrayIA[i], arrayVA[i], INSERT_VALUES);
		}
	}

	/*---------------------------------------------------------------------------
	 * assemble the matrix A
	 *---------------------------------------------------------------------------*/
	MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

	free(nnzA);
	free(arrayIA);
	free(arrayJA);
	free(arrayVA);
}

