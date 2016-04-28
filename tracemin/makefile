CPPFLAGS = -fopenmp

all: tracemin

include $(PETSC_DIR)/lib/petsc/conf/variables
include $(PETSC_DIR)/lib/petsc/conf/rules

tracemin_deflation: main.o TraceMin_Deflation.o Jacobi1.o QRFactorization.o matio.o mmio.o tracemin_cg_q1.o
	-${CLINKER} $^ $(PETSC_KSP_LIB) $(CPPFLAGS) -o $@

tracemin: main.o TraceMin.o Jacobi1.o QRFactorization.o matio.o mmio.o tracemin_cg_q1.o
	-${CLINKER} $^ $(PETSC_KSP_LIB) $(CPPFLAGS) -o $@
