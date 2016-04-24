include $(PETSC_DIR)/lib/petsc/conf/variables
include $(PETSC_DIR)/lib/petsc/conf/rules

tracemin : main.o TraceMin.o JacobiEigenDecomposition.o matio.o mmio.o
	-${CLINKER} $^ $(PETSC_KSP_LIB) -openmp -o $@
