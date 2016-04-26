/*************************************************************************
    > File Name: tracemin_cg.h
    > Author: xc
    > Descriptions: 
    > Created Time: Mon Apr 25 15:38:57 2016
 ************************************************************************/

#ifndef TRACE_MIN_CG
#define TRACE_MIN_CG

#include <petscconf.h>
#if defined(PETSC_HAVE_MATHIMF_H)
#include <mathimf.h>           /* this needs to be included before math.h */
#endif

#include <petscdt.h>            /*I "petscdt.h" I*/
#include <petscblaslapack.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/dtimpl.h>
#include <petscviewer.h>
#include <petscdmplex.h>
#include <petscdmshell.h>
#include <petscksp.h>
#include <petscmat.h>

class ProjectedMatrix
{
	public:
		ProjectedMatrix(Mat A, Mat Q1) : A_(A), Q1_(Q1) { }
		~ProjectedMatrix() { }

		static PetscErrorCode Mult(Mat PA_shell, Vec x, Vec y);

		Mat A_;
		Mat Q1_;
};

int tracemin_cg(Mat A,
                Mat X,
                Mat BY,
                Mat AY,
                PetscInt M,
                PetscInt N);

#endif // TRACE_MIN_CG
