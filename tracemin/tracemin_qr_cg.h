#ifndef TRACE_MIN_CG
#define TRACE_MIN_CG

#include <petscmat.h>

class ProjectedMatrix
{
	public:
		ProjectedMatrix(Mat A, Mat Q1) : A_(A), Q1_(Q1) { }
		~ProjectedMatrix() { }

		static PetscErrorCode MultVec(Mat PA_shell, Vec x, Vec y);
		static PetscErrorCode MultMat(Mat PA_shell, Mat X, Mat Y);

		Mat A_;
		Mat Q1_;
};

int tracemin_cg( Mat A, Mat X, Mat BY, Mat AY, PetscInt M, PetscInt N);
<<<<<<< HEAD:tracemin/tracemin_cg_q1.h
=======
int getQ1      ( Mat A, PetscInt M, PetscInt N); 

>>>>>>> 3e7d753d9f3c4a8ff6c295b2fc765fefe4116289:tracemin/tracemin_qr_cg.h

#endif // TRACE_MIN_CG
