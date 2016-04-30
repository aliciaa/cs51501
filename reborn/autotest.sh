MATRIX_A=${HOME}/cs51501/proj/testcases/A_100.mtx
#MATRIX_A=${HOME}/cs51501/proj/testcases/large/bloweybq.mtx
MATRIX_B=${HOME}/cs51501/proj/testcases/B_100.mtx
NPROC=1
LOWER_BOUND=0
UPPER_BOUND=1000
TRACEMIN_PROG=tracemin
MRES_OUTPUT=output.res

./${TRACEMIN_PROG} -p 4 -fA ${MATRIX_A} -fB ${MATRIX_B} -fO ${MRES_OUTPUT}
#mpirun -np ${NPROC} -machinefile hostfile.txt ./../multisection ${MATRIX_A} ${MATRIX_B} ${LOWER_BOUND} ${UPPER_BOUND} ${NPROC}

