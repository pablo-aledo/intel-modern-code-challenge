CXX=g++
COMPILER_VERSION := "$(CXX)-$(shell $(CXX) --version | head -n1 | cut -d' ' -f4)"
BUILD_HOST:=$(shell sh -c './BUILD-HOST-GEN')

CFLAGS = -DCOMPILER_VERSION=\"$(COMPILER_VERSION)\" -DBUILD_HOST=\"$(BUILD_HOST)\" -fopenmp

cell_clustering: cell_clustering.cpp util.cpp util.hpp Makefile
	$(CXX) -o $@ cell_clustering.cpp util.cpp $(CFLAGS) -Wall -lrt

clean:
	rm -rf cell_clustering run_clust.* output
	touch output
