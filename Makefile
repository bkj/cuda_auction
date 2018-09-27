# cuda_auction/Makefile

# ARCH=\
#   -gencode arch=compute_61,code=compute_61 \
#   -gencode arch=compute_61,code=sm_61

ARCH=\
  -gencode arch=compute_60,code=compute_60 \
  -gencode arch=compute_60,code=sm_60

OPTIONS=-O3 -use_fast_math

all: main shared cpu
	
main: src/auction.cu src/auction_kernel_dense.cu src/auction_kernel_csr.cu src/topdot.cpp
	mkdir -p bin
	nvcc $(ARCH) $(OPTIONS) --std=c++11 -o bin/auction src/auction.cu -I src -l curand

cpu: src/auction_cpu.cpp src/topdot.cpp
	mkdir -p bin
	g++ $(OPTIONS) -std=c++11 -lstdc++ -o bin/auction_cpu src/auction_cpu.cpp -I src

shared: src/auction.cu src/auction_kernel_dense.cu src/auction_kernel_csr.cu src/topdot.cpp
	mkdir -p lib
	nvcc $(ARCH) $(OPTIONS) --std=c++11 -Xcompiler -fPIC -shared -o lib/cuda_auction.so src/auction.cu -I src -l curand
	
clean:
	rm -rf bin lib