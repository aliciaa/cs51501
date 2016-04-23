#include "row_compressed_matrix.hpp"

int main() {
  RowCompressedMatrix A(8, 2, 0);
  RowCompressedMatrix B(8);
  int pos_eigen;
  int neg_eigen;
  A.dump();
  B.dump();
  A.count_eigen(pos_eigen, neg_eigen); 
  RowCompressedMatrix C = a_plus_mu_b(A, 2.0, B);
  C.dump();
  C.count_eigen(pos_eigen, neg_eigen);
}
