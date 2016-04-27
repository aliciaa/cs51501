#include <stdio.h>
#include <iostream>
#include "mpi.h"
#include "row_compressed_matrix.hpp"
#include "sys/time.h"

#define MPI_MASTER 0
#define SPLIT_PER_PROC 4

#define UNEVEN_RELATIVE_TOL 0.1
#define UNEVEN_ABSOLUTE_TOL 5

#define MAX_LIST_SIZE 1000

class LinearList {
 public:
  LinearList() {
    _len = 0;
    _iter = 0;
  }

  void insert(double pos, int value) {
    int l = 0;
    while (l < _len && _pos[l] < pos) {
      l++;
    }
    for (int i = _len; i > l; i--) {
      _pos[i] = _pos[i-1];
      _value[i] = _value[i-1];
    }
    _pos[l] = pos;
    _value[l] = value;
    _len++;
  }

  bool reset_iterator() {
    _iter = 0;
    return _iter < _len;
  }

  bool get_next(double& pos, int& value) {
    pos = _pos[_iter];
    value = _value[_iter];
    _iter++;
    return _iter < _len;
  }

 private:
  double _pos[MAX_LIST_SIZE];
  int _value[MAX_LIST_SIZE];
  int _len;
  int _iter;
};

/*
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
	curr_eigs = (splitters[i+1] - intervals[l]) * local_density;
	l++;
      } else {
        curr_eigs += num_eigs[i+1] - num_eigs[i];
      }
    }
  }
}
*/


void pardiso_worker(const RowCompressedMatrix& A,
                    const RowCompressedMatrix& B,
		    double lower_bound, double upper_bound,
		    int rank, int nproc,
		    bool contain_boundary,
		    double* splitters, int* num_eigs) {
  int local_eigs[SPLIT_PER_PROC];

  //split to sub-sub intervals
  for (int i = 0; i < SPLIT_PER_PROC * nproc; i++) {
    if (contain_boundary) {
      splitters[i] = lower_bound + (upper_bound - lower_bound)*i / (SPLIT_PER_PROC*nproc-1);
    } else {
      splitters[i] = lower_bound + (upper_bound - lower_bound) * (i+1) / (SPLIT_PER_PROC * nproc + 1);
    }
    if (rank == MPI_MASTER) {
      //printf("splitters[%d] = %.5f\n", i, splitters[i]);
    }
  }
  int pos_eigen;
  int neg_eigen;
  //parallel computing num of eigs in sub-sub intervals of each sub intervals
  for (int i = 0; i < SPLIT_PER_PROC; i++) {
    RowCompressedMatrix C = a_plus_mu_b(A, -splitters[rank*SPLIT_PER_PROC+i], B);
    C.count_eigen(pos_eigen, neg_eigen);
    local_eigs[i] = neg_eigen;
  }
  MPI_Allgather(local_eigs, SPLIT_PER_PROC, MPI_INT, num_eigs, SPLIT_PER_PROC, MPI_INT, MPI_COMM_WORLD);

}

bool check(LinearList& eig_list, double& v_lower, double& v_upper, double average_eigs) {
  bool has_next = eig_list.reset_iterator();
  if (!has_next) {return false;}
  double prev_pos, curr_pos;
  int prev_eig, curr_eig;
  has_next = eig_list.get_next(prev_pos, prev_eig);
  while (has_next) {
    has_next = eig_list.get_next(curr_pos, curr_eig);
    if ((curr_eig - prev_eig > UNEVEN_ABSOLUTE_TOL) &&
        ((curr_eig - prev_eig) / average_eigs > UNEVEN_RELATIVE_TOL)) {
      v_lower = prev_pos;
      v_upper = curr_pos;
      return true;
    }
    prev_pos = curr_pos;
    prev_eig = curr_eig;
  }
  return false;
}

void get_multisection(const RowCompressedMatrix& A,
                      const RowCompressedMatrix& B,
		      double lower_bound, double upper_bound,
		      int rank, int nproc,
		      int num_intervals, double* intervals) {
  double* splitters = new double[nproc * SPLIT_PER_PROC];
  int* num_eigs = new int[nproc * SPLIT_PER_PROC];
  pardiso_worker(A, B, lower_bound, upper_bound, rank, nproc,
                 true, splitters, num_eigs);
  int rounds = 1;
  LinearList eig_list;
  for (int i = 0; i < nproc * SPLIT_PER_PROC; i++) {
    //if (rank == MPI_MASTER) {
      //printf("pos = %.5f, neg_eigs = %d\n", splitters[i], num_eigs[i]);
      //fflush(stdout);
    //}
    eig_list.insert(splitters[i], num_eigs[i]);
  }
  double total_eigs = num_eigs[nproc * SPLIT_PER_PROC - 1] - num_eigs[0];
  double average_eigs = total_eigs / num_intervals;
  double v_lower = 0;
  double v_upper = 0;;
  while (check(eig_list, v_lower, v_upper, average_eigs)) {
    rounds++;
    pardiso_worker(A, B, v_lower, v_upper, rank, nproc,
                   false, splitters, num_eigs);
    for (int i = 0; i < nproc * SPLIT_PER_PROC; i++) {
      eig_list.insert(splitters[i], num_eigs[i]);
    }
  }

  if (rank == MPI_MASTER) {
    double curr_eigs = 0;
    double prev_pos, curr_pos;
    int prev_num, curr_num;
    eig_list.reset_iterator();
    bool has_next = eig_list.get_next(prev_pos, prev_num);
    int l = 1;
    while (l < num_intervals && has_next) {
      has_next = eig_list.get_next(curr_pos, curr_num);
      if (curr_num - prev_num + curr_eigs > average_eigs) {
        if (average_eigs - curr_eigs > curr_num - prev_num + curr_eigs - average_eigs) {
	  intervals[l] = curr_pos;
	  average_eigs = (average_eigs*(num_intervals-l+1) - (curr_num-prev_num+curr_eigs)) / (num_intervals - l);
	  curr_eigs = 0;
	  l++;

	} else {
	  intervals[l] = prev_pos;
	  average_eigs = (average_eigs*(num_intervals-l+1) - curr_eigs) / (num_intervals - l);
	  curr_eigs = curr_num - prev_num;
	  l++;
	}
      } else {
        curr_eigs += curr_num - prev_num;
      }
      prev_pos = curr_pos;
      prev_num = curr_num;
    }
    std::cout << "Total rounds = " << rounds << std::endl;
    std::cout << "Total LDL = " << rounds * SPLIT_PER_PROC << std::endl;
  }
}

//int main(int argc, char* argv[]) {
void multisection(int argc, char* argv[],
                  const char* file_A, const char* file_B,
                  double lower_bound, double upper_bound, int num_intervals) {
  MPI_Init(&argc, &argv);
  int nproc, rank;
  char pname[100];
  int plen;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Get_processor_name(pname, &plen);
  printf("I'm rank %d in %d on %s\n", rank, nproc, pname); 
  RowCompressedMatrix A(file_A);
  RowCompressedMatrix B(file_B);
  // Read Matrix A and B on each node
  //RowCompressedMatrix A(10000, 10, 0);
  //RowCompressedMatrix B(10000);
  //RowCompressedMatrix A("bcsstk01.mtx");
  //RowCompressedMatrix B(48);
  //double lower_bound = 0;
  //double upper_bound = 200;
  //int num_intervals = 4;
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
  if (rank == MPI_MASTER) {
    printf("multisectioning cost %.6f seconds.\n", time_elapsed);
  }


  intervals[num_intervals] = upper_bound;
  if (rank == MPI_MASTER) {
    for (int i = 0; i <= num_intervals; i++) {
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
    FILE* fp = fopen("interval_file.txt", "w");
    fprintf(fp, "%d\n", num_intervals);
    for (int i = 0; i <= num_intervals; i++) {
      fprintf(fp, "%.8lf\n", intervals[i]);
    }
    for (int i = 0; i < num_intervals; i++) {
      fprintf(fp, "%d\n", num_eigs[i+1] - num_eigs[i]);
    }
    fclose(fp);
  }
  MPI_Finalize();
}

int main(int argc, char* argv[]){
  if (argc <= 5) {
    printf("USAGE : ./multisection A.mtx B.mtx lower upper num_intervals\n");
    exit(-1);
  }
  const char* file_A = argv[1];
  const char* file_B = argv[2];
  double lower_bound = atof(argv[3]);
  double upper_bound = atof(argv[4]);
  int num_intervals = atoi(argv[5]);
  printf("A<%s> B<%s> <%.10f, %.10f> = <%d>\n", file_A, file_B, lower_bound, upper_bound, num_intervals);
  multisection(argc, argv, file_A, file_B, lower_bound, upper_bound, num_intervals);
  //RowCompressedMatrix a("bcsstk01.mtx");
}
