CXX=mpiicc


FLAGS=-DMKL_ILP64 -qopenmp -I${MKLROOT}/include

#DFLAGS= -g

all : multisection

multisection : multisection.o row_compressed_matrix.o mmio.o
	$(CXX) $(DFLAGS) --std=c++11 -I${MKLROOT}/include $^ -o $@ -L . -lpardiso500-INTEL1301-X86-64 -L${MKLROOT}/lib/intel64 -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lm -lifcore -fopenmp

eigen_counter.o : eigen_counter.cpp
	$(CXX) $(DFLAGS) --std=c++11 $^ -c -o $@

multisection.o : multisection.cpp
	$(CXX) $(DFLAGS) --std=c++11 $^ -c -o $@

mmio.o : mmio.cpp
	$(CXX) $(DFLAGS) --std=c++11 $^ -c -o $@

row_compressed_matrix.o: row_compressed_matrix.cpp row_compressed_matrix.hpp
	$(CXX) $(DFLAGS) --std=c++11 row_compressed_matrix.cpp -c -o $@

clean:
	rm *.o
