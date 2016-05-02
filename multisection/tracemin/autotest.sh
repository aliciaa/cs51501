MATRIX_A=${HOME}/cs51501/proj/testcases/A_100.mtx
MATRIX_B=${HOME}/cs51501/proj/testcases/B_100.mtx
#MATRIX_A=${HOME}/cs51501/proj/testcases/large/bloweybq.mtx
#MATRIX_A=${HOME}/cs51501/proj/testcases/superlarge/x104.mtx
#MATRIX_B=${HOME}/cs51501/proj/testcases/B_108384.mtx
NPROC=1
LOWER_BOUND=0
UPPER_BOUND=1000
TRACEMIN_PROG=tracemin

MRES_OUTPUT=eigen_mres

rm *.o tracemin tracemin_deflation
make -s

mpirun -np ${NPROC} -machinefile hostfile.txt ./${TRACEMIN_PROG} -fA ${MATRIX_A} -fB ${MATRIX_B} -fO ${MRES_OUTPUT}
