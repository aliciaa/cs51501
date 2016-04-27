MATRIX_A=${HOME}/cs51501/proj/testcases/A_tiny.mtx
MATRIX_B=${HOME}/cs51501/proj/testcases/B_tiny.mtx
NPROC=4
LOWER_BOUND=0
UPPER_BOUND=100

mpirun -np ${NPROC} -machinefile hostfile.txt ./../multisection ${MATRIX_A} ${MATRIX_B} ${LOWER_BOUND} ${UPPER_BOUND} ${NPROC}

echo "#define CG" >> linear_solver.h
echo "##################################"
echo "Testing using CG Linear Solver"
echo "##################################"

CG_OUTPUT=eigen_cg

mpirun -np ${NPROC} -machinefile hostfile.txt ./tracemin -fA ${MATRIX_A} -fB ${MATRIX_B} -fO ${CG_OUTPUT}

echo "#define MINRES" >> linear_solver.h
echo "##################################"
echo "Testing using MINRES Linear Solver"
echo "##################################"

MRES_OUTPUT=eigen_mres

mpirun -np ${NPROC} -machinefile hostfile.txt ./tracemin -fA ${MATRIX_A} -fB ${MATRIX_B} -fO ${MRES_OUTPUT}
