MATRIX_A=${HOME}/cs51501/proj/testcases/A_100.mtx
#MATRIX_A=${HOME}/cs51501/proj/testcases/large/bloweybq.mtx
MATRIX_B=${HOME}/cs51501/proj/testcases/B_100.mtx
NPROC=1
LOWER_BOUND=0
UPPER_BOUND=1000
TRACEMIN_PROG=tracemin_davidson
MRES_OUTPUT=output.res

#mpirun -np ${NPROC} ./../multisection/multisection ${MATRIX_A} ${MATRIX_B} ${LOWER_BOUND} ${UPPER_BOUND} ${NPROC}

mpirun -np ${NPROC} ./${TRACEMIN_PROG} -p 4 -fA ${MATRIX_A} -fB ${MATRIX_B} -fO ${MRES_OUTPUT}

