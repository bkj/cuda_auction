
# Generate SASS for the first version of each major architecture.
#   This will cover that entire major architecture.
# Generate SASS for important minor versions.
# Generate PTX for the last named architecture for future support.
ARCH=\
  -gencode arch=compute_61,code=compute_61 \
  -gencode arch=compute_61,code=sm_61

OPTIONS=-O3 -use_fast_math

all: main
	
main: main.cu auction_kernel.cu
	nvcc $(ARCH) $(OPTIONS) -o main main.cu

clean:
	rm -f main