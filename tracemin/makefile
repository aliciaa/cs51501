EXEC = tracemin

all: tracemin

include $(PETSC_DIR)/lib/petsc/conf/variables
include $(PETSC_DIR)/lib/petsc/conf/rules

$(EXEC): main.o TraceMin.o JacobiEigenDecomposition.o matio.o mmio.o tracemin_qr.o tracemin_cg.o
	-${CLINKER} $^ $(PETSC_KSP_LIB) -fopenmp -o $@


testcg: testcg.o
	-${CLINKER} $^ $(PETSC_KSP_LIB) -fopenmp -o $@

