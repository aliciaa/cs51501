# source files
CPPSRC = $(wildcard *.cpp)
CPPOBJ = $(CPPSRC:%.cpp=%.o)
CSRC = $(wildcard *.c)
COBJ = $(CSRC:%.c=%.o)
SRC = $(CPPSRC) $(CSRC)
OBJ = $(CPPOBJ) $(COBJ)
EXE = tracemin tracemin_deflation tracemin_davidson

# compiler
COMPILE = mpiicpc

# include paths
INCLUDES = -I. -I${MKLROOT}/include

# compile flags
CCFLAGS = -DMKL_ILP64 -qopenmp -O3 --std=c++11

# link flags
LDFLAGS = ${MKLROOT}/lib/intel64/libmkl_scalapack_ilp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_ilp64.a -Wl,--end-group -lpthread -lm -liomp5

.PHONY: clean

all: $(EXE)

%.o : %.cpp
	$(COMPILE) $(INCLUDES) $(CCFLAGS) -c $< -o $@

%.o : %.c
	$(COMPILE) $(INCLUDES) $(CCFLAGS) -c $< -o $@

tracemin_deflation: main.o view.o matio.o mmio.o tracemin_deflation.o jacobi.o linear_solver_mres.o
	$(COMPILE) $^ $(LDFLAGS) -o $@

tracemin_davidson: main.o view.o matio.o mmio.o tracemin_davidson.o jacobi.o linear_solver_cg.o
	$(COMPILE) $^ $(LDFLAGS) -o $@

tracemin: main.o view.o matio.o mmio.o tracemin.o jacobi_o.o linear_solver_mres.o
	$(COMPILE) $^ $(LDFLAGS) -o $@

clean:
	rm -f $(OBJ) $(EXE)

