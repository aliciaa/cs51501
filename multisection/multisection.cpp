#include <stdio.h>
#include <iostream>
#include "mpi.h"
#include "row_compressed_matrix.hpp"
#include "sys/time.h"

#define MPI_MASTER 0
#define SPLIT_PER_PROC 10

void get_multisection(const RowCompressedMatrix& A,
                      const RowCompressedMatrix& B,
		      double lower_bound, double upper_bound,
		      int rank, int nproc,
		      int num_intervals, double* intervals) {

  int pos_eigen;
  int neg_eigen;
  double* splitters = new double[nproc * SPLIT_PER_PROC];
  int* num_eigs = new int[nproc * SPLIT_PER_PROC];
  int* local_eigs = new int[SPLIT_PER_PROC];

  //split to sub-sub intervals
  for (int i = 0; i < SPLIT_PER_PROC * nproc; i++) {
     splitters[i] = lower_bound + (upper_bound - lower_bound)*i / (SPLIT_PER_PROC*nproc-1);
  }
  //parallel computing num of eigs in sub-sub intervals of each sub intervals
  for (int i = 0; i < SPLIT_PER_PROC; i++) {
    RowCompressedMatrix C = a_plus_mu_b(A, -splitters[rank*SPLIT_PER_PROC+i], B);
    C.count_eigen(pos_eigen, neg_eigen);
    local_eigs[i] = neg_eigen;
  }
  MPI_Gather(local_eigs, SPLIT_PER_PROC, MPI_INT, num_eigs, SPLIT_PER_PROC, MPI_INT, MPI_MASTER, MPI_COMM_WORLD);
  if (rank == MPI_MASTER) {
    for (int i = 0; i < SPLIT_PER_PROC * nproc; i++) {
      //std::cout << "splitter = " << splitters[i] << std::endl;
      //std::cout << "neg_eigs = " << num_eigs[i] << std::endl;
    }
    int l = 1;
    double curr_eigs = 0;
    double expect_avg_eigs = double((num_eigs[SPLIT_PER_PROC*nproc - 1] - num_eigs[0])) / num_intervals;
    printf("avg_eigs = %.5f\n", expect_avg_eigs);
    for (int i = 0; i < SPLIT_PER_PROC * nproc - 1; i++) {
      if (curr_eigs + num_eigs[i+1] - num_eigs[i] >= expect_avg_eigs) {
        //printf("l = %d, curr_eigs = %.5lf \n", l, curr_eigs);
	//printf("i = %d, num_eigs[i+1] = %d, num_eigs[i] = %d\n", i, num_eigs[i+1], num_eigs[i]);
	//printf("splitters[i+1] = %.5f, splitters[i] = %.5f\n", splitters[i+1], splitters[i]);
        double local_density = (num_eigs[i+1] - num_eigs[i]) / ((upper_bound - lower_bound) / (SPLIT_PER_PROC * nproc - 1));
	//printf("local_density = %.5f\n", local_density);
	intervals[l] = splitters[i] + (expect_avg_eigs - curr_eigs) / local_density;
	//why? why not curr_eigs+=
	curr_eigs = (splitters[i+1] - intervals[l]) * local_density;
	l++;
      } else {
        //why not need to store intervals[]?
        curr_eigs += num_eigs[i+1] - num_eigs[i];
      }
    }
  }
}

//how to control error?
//how to test time?

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int nproc, rank;
  char pname[100];
  int plen;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Get_processor_name(pname, &plen);
  printf("I'm rank %d in %d on %s\n", rank, nproc, pname); 
  // Read Matrix A and B on each node
  RowCompressedMatrix A(100000, 10, 0);
  RowCompressedMatrix B(100000);
  //RowCompressedMatrix A(1000, 2, 0);
  //RowCompressedMatrix B(1000);
  double lower_bound = 0;
  double upper_bound = 200;
  int num_intervals = 4;
  double* intervals = new double[num_intervals+1];
  intervals[0] = lower_bound;

  struct timeval tv1, tv2;
  double time_elapsed = 0;
  gettimeofday(&tv1, NULL);

  if (nproc != 1) {
    get_multisection(A, B, lower_bound, upper_bound, rank, nproc, num_intervals, intervals);
  }
  gettimeofday(&tv2, NULL);
  time_elapsed = (tv2.tv_sec - tv1.tv_sec) +
                 (tv2.tv_usec - tv1.tv_usec) / 1000000.0;
  printf("multisectioning cost %.6f seconds.\n", time_elapsed);


  intervals[num_intervals] = upper_bound;
  if (rank == MPI_MASTER) {
    for (int i = 0; i <= nproc; i++) {
      std::cout << "intervals["<<i<<"] = " << intervals[i] << std::endl;
    }
    int* num_eigs = new int[num_intervals+1];
    for (int i = 0; i <= num_intervals; i++) {
      RowCompressedMatrix C = a_plus_mu_b(A, -intervals[i], B);
      int pos_eigs, neg_eigs;
      C.count_eigen(pos_eigs, neg_eigs);
      num_eigs[i] = neg_eigs;
    }
    printf("Total egis on interval [%.5f, %.5f] = %d \n", lower_bound, upper_bound, num_eigs[num_intervals] - num_eigs[0]);
    for (int i = 0; i < num_intervals; i++) {
      printf("Number of eigs on interval [%.5f, %.5f] = %d\n", intervals[i], intervals[i+1], num_eigs[i+1]-num_eigs[i]);
    }
  }
/*  
  int pos_eigen;
  int neg_eigen;
  A.dump();
  B.dump();
  A.count_eigen(pos_eigen, neg_eigen); 
  RowCompressedMatrix C = a_plus_mu_b(A, 2.0, B);
  C.dump();
  C.count_eigen(pos_eigen, neg_eigen);
*/
  MPI_Finalize();
}
