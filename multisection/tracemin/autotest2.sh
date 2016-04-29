#MATRIX_A=${HOME}/cs51501/proj/testcases/A_tiny.mtx
MATRIX_A=${HOME}/cs51501/proj/testcases/large/bloweybq.mtx
MATRIX_B=${HOME}/cs51501/proj/testcases/B_tiny03.mtx
NPROC=2
LOWER_BOUND=0
UPPER_BOUND=400000

mpirun -np ${NPROC} -machinefile hostfile.txt ./../multisection ${MATRIX_A} ${MATRIX_B} ${LOWER_BOUND} ${UPPER_BOUND} ${NPROC}

rm linear_solver.h
echo "#define CG" >> linear_solver.h
echo "##################################"
echo "Testing using CG Linear Solver"
echo "##################################"

CG_OUTPUT=eigen_cg

rm *.o tracemin
make -s

mpirun -np ${NPROC} -machinefile hostfile.txt ./tracemin -fA ${MATRIX_A} -fB ${MATRIX_B} -fO ${CG_OUTPUT}

rm linear_solver.h
echo "#define MINRES" >> linear_solver.h
echo "##################################"
echo "Testing using MINRES Linear Solver"
echo "##################################"

MRES_OUTPUT=eigen_mres

rm *.o tracemin
make -s

mpirun -np ${NPROC} -machinefile hostfile.txt ./tracemin -fA ${MATRIX_A} -fB ${MATRIX_B} -fO ${MRES_OUTPUT}
