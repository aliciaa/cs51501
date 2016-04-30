CXX=icc

MKL_MIC_ENABLE=1

MKL=${MKLROOT}/lib/intel64/libmkl_scalapack_ilp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_ilp64.a -Wl,--end-group -lpthread -lm

FLAGS=-DMKL_ILP64 -qopenmp -I${MKLROOT}/include


all : linear_solver_mres

linear_solver_mres : linear_solver_mres.o
	$(CXX) $(FLAGS) -std=c++11 $^ -o $@ $(MKL)

linear_solver_mres.o : linear_solver_mres.cpp
	$(CXX) $(FLAGS) -std=c++11 $^ -c -o $@

clean :
	rm *.o linear_solver_mres

