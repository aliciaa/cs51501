#ifndef ROW_COMPRESSED_MATRIX_H_
#define ROW_COMPRESSED_MATRIX_H_

#include <iostream>
#include <stdio.h>

// A Wrapper for compressed sparse row format symmetric matrix

class RowCompressedMatrix {
 public:
  // The default constructor will give a test matrix;
  RowCompressedMatrix() {
    _num_rows = 8;
    _num_nnz = 18;
    _allocate_memory();
    _c_index_style = true;
    _ia[0] = 0; _ia[1] = 4; _ia[2] = 7;
    _ia[3] = 9; _ia[4] = 11; _ia[5] = 14;
    _ia[6] = 16; _ia[7] = 17; _ia[8] = 18;

    _ja[0] = 0; _ja[1] = 2; _ja[2] = 5;
    _ja[3] = 6; _ja[4] = 1; _ja[5] = 2;
    _ja[6] = 4; _ja[7] = 2; _ja[8] = 7;
    _ja[9] = 3; _ja[10] = 6; _ja[11] = 4;
    _ja[12] = 5; _ja[13] = 6; _ja[14] = 5;
    _ja[15] = 7; _ja[16] = 6; _ja[17] = 7;

    _a[0] = 7; _a[1] = 1; _a[2] = 2;
    _a[3] = 7; _a[4] = -4; _a[5] = 8;
    _a[6] = 2; _a[7] = 1; _a[8] = 5;
    _a[9] = 7; _a[10] = 9; _a[11] = 5;
    _a[12] = -1; _a[13] = 5; _a[14] = 0;
    _a[15] = 5; _a[16] = 11; _a[17] = 5;
  }
  // A tri-diagonal matrix of {-1, 4, -1}
  RowCompressedMatrix(int num_rows) {
    _num_rows = num_rows;
    _num_nnz = num_rows * 2 - 1;
    _allocate_memory();
    _c_index_style = true;

    int count_nnz = 0;
    for (int i = 0; i < _num_rows; i++) {
      _ia[i] = count_nnz;
      _ja[count_nnz] = i;
      _a[count_nnz] = 4;
      count_nnz++;
      if (i != _num_rows - 1) {
        _ja[count_nnz] = i + 1;
        _a[count_nnz] = -1;
	count_nnz++;
      }
    }
    _ia[_num_rows] = count_nnz;
  }

  RowCompressedMatrix(char const* file_name);

  //allcate memory for the matrix
  RowCompressedMatrix(int num_rows, int num_nnz)
    : _num_rows(num_rows), _num_nnz(num_nnz) {_allocate_memory();}
 
  //generate A myself
  RowCompressedMatrix(int num_rows, int bandwidth, int dummy)
    : _num_rows(num_rows) {
    _num_nnz = num_rows * bandwidth - (bandwidth * (bandwidth-1)) / 2;
    _allocate_memory();
    std::cout << "_num_rows = " << _num_rows << std::endl;
    std::cout << "_num_nnz = " << _num_nnz << std::endl;
    int curr_count = 0;
    for (int i = 0; i <= _num_rows; i++) {
      _ia[i] = curr_count;
      for (int j = i; j < (i+bandwidth) && j < _num_rows; j++) {
        if (j == i) {
	//  _a[curr_count] = 100;
	  _a[curr_count] = 100;
	} else {
	  _a[curr_count] = j-i;
	}
	_ja[curr_count] = j;
	//printf("curr_count = %d, i = %d, j = %d \n", curr_count, i, j);
	curr_count++;
      }
    }
    _ia[_num_rows] = curr_count;    
  }


  RowCompressedMatrix(const RowCompressedMatrix& rhs) {
    _num_rows = rhs._num_rows;
    _num_nnz = rhs._num_nnz;
    _allocate_memory();
    for (int i = 0; i <= _num_rows; i++) {
      _ia[i] = rhs._ia[i];
    }
    for (int i = 0; i < _num_nnz; i++) {
      _ja[i] = rhs._ja[i];
      _a[i] = rhs._a[i];
    }
  }
  ~RowCompressedMatrix() {
    delete[] _ia;
    delete[] _ja;
    delete[] _a;
  }

  void count_eigen(int& pos_eigen, int& neg_eigen);

  void dump() const;

 private:
  void _allocate_memory() {
    _c_index_style = true;
    _ia = new int[_num_rows + 1];
    _ja = new int[_num_nnz];
    _a = new double[_num_nnz];
  }
  void _to_fortran_index();
  void _to_c_index();
  // number of rows
  int _num_rows;
  // number of non-zeros
  int _num_nnz;
  // row index begin
  int* _ia;
  // column indices
  int* _ja;
  // values
  double* _a;
  // true : c-index, 0 based; false : fortran-index, 1 based.
  bool _c_index_style;

  friend RowCompressedMatrix a_plus_mu_b(const RowCompressedMatrix&,
                                         double,
					 const RowCompressedMatrix&);
};

RowCompressedMatrix a_plus_mu_b(const RowCompressedMatrix& A,
                                double mu,
                                const RowCompressedMatrix& B);

#endif // ROW_COMPRESSED_MATRIX_H_
