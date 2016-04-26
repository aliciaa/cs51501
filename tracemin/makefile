CPPFLAGS = -openmp

all: tracemin

include $(PETSC_DIR)/lib/petsc/conf/variables
include $(PETSC_DIR)/lib/petsc/conf/rules

tracemin: main.o TraceMin.o JacobiEigenDecomposition.o QRFactorization.o matio.o mmio.o tracemin_cg_q1.o
	-${CLINKER} $^ $(PETSC_KSP_LIB) $(CPPFLAGS) -o $@
