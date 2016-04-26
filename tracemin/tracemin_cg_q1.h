#ifndef TRACE_MIN_CG
#define TRACE_MIN_CG

#include <petscmat.h>

class ProjectedMatrix
{
	public:
		ProjectedMatrix(Mat A, Mat Q1) : A_(A), Q1_(Q1) { }
		~ProjectedMatrix() { }

		static PetscErrorCode MultVec(Mat PA_shell, Vec x, Vec y);

		Mat A_;
		Mat Q1_;
};


PetscErrorCode tracemin_cg(const Mat A, Mat X, const Mat BY, const Mat AY, PetscInt M, PetscInt N);

#endif // TRACE_MIN_CG
