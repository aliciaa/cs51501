# Petsc Readme for TraceMin alg of CS51501

* install PETSc under mc cluster machine

 1. First Download Petsc from http://www.mcs.anl.gov/petsc/download/  
 2. untar the file.  
 3. $./configure --with-cc=gcc --with-cxx=g++ --with-fc=gfortran --download-fblaslapack --download-mpich  
 4. $make all test


＊ configure the environment
added 
export PETSC_DIR=/homes/cheng172/soft/petsc/petsc-3.6.3                     
export PETSC_ARCH=linux-gnu-c-debug
to .bashrc

* Add the code to the project folder
move tracemin.c to ~/petsc/petsc-3.6.3/src/ksp/ksp/examples/tutorials/
go to ~/petsc/petsc-3.6.3/src/ksp/ksp/example/tutorials/

added 

  tracemin: tracemin.o chkopts  
    -${CLINKER} -o tracemin tracemin.o  ${PETSC_KSP_LIB}  
    ${RM} tm.o  

to makefile

make tracemin


