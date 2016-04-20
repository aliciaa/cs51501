* Petsc is stupid!

Download Petsc from http://www.mcs.anl.gov/petsc/download/
untar the file.
  ./configure --with-cc=gcc --with-cxx=g++ --with-fc=gfortran --download-fblaslapack --download-mpich
  make all test

added 

export PETSC_DIR=/homes/cheng172/soft/petsc/petsc-3.6.3                     
export PETSC_ARCH=linux-gnu-c-debug

to .bashrc

move tracemin.c to ~/petsc/petsc-3.6.3/src/ksp/ksp/examples/tutorials/
go to ~/petsc/petsc-3.6.3/src/ksp/ksp/example/tutorials/

added 

  tracemin: tracemin.o chkopts
    -${CLINKER} -o tracemin tracemin.o  ${PETSC_KSP_LIB}
    ${RM} tm.o

to makefile

make tracemin
