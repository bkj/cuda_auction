# cuda_auction/Makefile

# ARCH=\
#   -gencode arch=compute_61,code=compute_61 \
#   -gencode arch=compute_61,code=sm_61

ARCH=\
  -gencode arch=compute_60,code=compute_60 \
  -gencode arch=compute_60,code=sm_60

OPTIONS=-O3 -use_fast_math

all: main shared
	
main: src/auction.cu src/auction_kernel_dense.cu src/auction_kernel_csr.cu
	mkdir -p bin
	nvcc $(ARCH) $(OPTIONS) -o bin/auction src/auction.cu -I src

shared: src/auction.cu src/auction_kernel_dense.cu src/auction_kernel_csr.cu
	mkdir -p lib
	nvcc $(ARCH) $(OPTIONS) -Xcompiler -fPIC -shared -o lib/cuda_auction.so src/auction.cu -I src
	
clean:
	rm -rf bin lib