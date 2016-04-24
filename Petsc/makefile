include $(PETSC_DIR)/lib/petsc/conf/variables
include $(PETSC_DIR)/lib/petsc/conf/rules

tracemin : JacobiEigenDecomposition.o matio.o mmio.o
	-${CLINKER} $^ $(PETSC_KSP_LIB) -openmp -o $@
