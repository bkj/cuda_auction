// auction.cu
//
// !! For best performance, I think the datalayout
// needs to be transposed.  Eg:
//      i + num_nodes * j
// instead of the current
//      i * num_nodes + j

#ifndef MAIN_AUCTION
#define MAIN_AUCTION

#include <iostream>
#include <string>
#include <fstream>
#include <chrono>

#include <stdio.h>
#include <stdlib.h>

#include <curand.h>
#include <curand_kernel.h>

// --
// Define constants


#ifndef __RUN_VARS
#define __RUN_VARS
#define MAX_NODES       20000 // Dimension of problem
#define BLOCKSIZE       32 // How best to set this?
#define AUCTION_MAX_EPS 1.0 // Larger values mean solution is more approximate
#define AUCTION_MIN_EPS 1.0
#define AUCTION_FACTOR  0.0
#define NUM_RUNS        10

// Uncomment to run dense version
// #define DENSE
#endif

#include "topdot.cpp"
#ifdef DENSE
    #include "auction_kernel_dense.cu"
#else
    #include "auction_kernel_csr.cu"
#endif


int load_data(float *raw_data) {
    std::ifstream input_file("graph", std::ios_base::in);

    std::cerr << "load_data: start" << std::endl;
    int i = 0;
    float val;
    while(input_file >> val) {
        raw_data[i] = val;
        i++;
        if(i > MAX_NODES * MAX_NODES) {
            std::cerr << "load_data: ERROR -- data file too large" << std::endl;
            return -1;
        }
    }
    std::cerr << "load_data: finish" << std::endl;
    return (int)sqrt(i);
}

extern "C" {

int run_auction(
    int    num_nodes,
    int    num_edges,

    float* h_data,      // data
    int*   h_offsets,   // offsets for items
    int*   h_columns,

    int*   h_person2item, // results

    float auction_max_eps,
    float auction_min_eps,
    float auction_factor,

    int num_runs,
    int verbose
)
{
    // --
    // CUDA options

    dim3 dimBlock(BLOCKSIZE, 1, 1);
    int gx = ceil(num_nodes / (double) dimBlock.x);
    dim3 dimGrid(gx, 1, 1);

    // --
    // Declare variables

    float* d_data;
    int*   d_offsets;
    int*   d_columns;

    int* d_person2item;
    int* d_item2person;

    float* d_bids;
    float* d_prices;
    int*   d_bidders; // unused
    int*   d_sbids;

    int  h_numAssign;
    int* d_numAssign = 0;
    float* d_rand;

    // --
    // Allocate device memory
    cudaMalloc((void **)&d_data,    num_edges * sizeof(float));
    cudaMalloc((void **)&d_columns, num_edges * sizeof(float));
    cudaMalloc((void **)&d_offsets, (num_nodes + 1) * sizeof(int));

    cudaMalloc((void **)&d_person2item, num_nodes * sizeof(int));
    cudaMalloc((void **)&d_item2person, num_nodes * sizeof(int));

    cudaMalloc((void **)&d_bids,    num_nodes * num_nodes * sizeof(float));
    cudaMalloc((void **)&d_prices,  num_nodes * sizeof(float));
    cudaMalloc((void **)&d_bidders, num_nodes * num_nodes * sizeof(int)); // unused
    cudaMalloc((void **)&d_sbids,   num_nodes * sizeof(int));

    cudaMalloc((void **)&d_numAssign, 1 * sizeof(int)) ;
    cudaMalloc((void **)&d_rand,      num_nodes * num_nodes * sizeof(float)) ;

    // --
    // Copy from host to device

    cudaMemcpy(d_data,    h_data,    num_edges       * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_columns, h_columns, num_edges       * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets, (num_nodes + 1) * sizeof(int),   cudaMemcpyHostToDevice);

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 123);
    curandGenerateUniform(gen, d_rand, num_nodes * num_nodes);

    for(int run_num = 0; run_num < num_runs; run_num++) {

        cudaMemset(d_prices, 0.0, num_nodes * sizeof(float));

        // Start timer
        cudaEvent_t start, stop;
        float milliseconds = 0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        float auction_eps = auction_max_eps;
        while(auction_eps >= auction_min_eps) {
            h_numAssign = 0;
            cudaMemset(d_bidders,        0, num_nodes * num_nodes * sizeof(int)); // unused
            cudaMemset(d_person2item,   -1, num_nodes * sizeof(int));
            cudaMemset(d_item2person,   -1, num_nodes * sizeof(int));
            cudaMemset(d_numAssign,      0, 1         * sizeof(int));
            cudaThreadSynchronize();

            int counter = 0;
            while(h_numAssign < num_nodes){
                counter += 1;
                cudaMemset(d_bids,  0, num_nodes * num_nodes * sizeof(float));
                cudaMemset(d_sbids, 0, num_nodes * sizeof(int));
                cudaThreadSynchronize();

                run_bidding<<<dimBlock, dimGrid>>>(
                    num_nodes,

                    d_data,
                    d_offsets,
                    d_columns,

                    d_person2item,
                    d_bids,
                    d_bidders,
                    d_sbids,
                    d_prices,
                    auction_eps,
                    d_rand
                );
                run_assignment<<<dimBlock, dimGrid>>>(
                    num_nodes,
                    d_person2item,
                    d_item2person,
                    d_bids,
                    d_bidders,
                    d_sbids,
                    d_prices,
                    d_numAssign
                );
                cudaThreadSynchronize();

                cudaMemcpy(&h_numAssign, d_numAssign, sizeof(int) * 1, cudaMemcpyDeviceToHost);
                // std::cerr << "h_numAssign=" << h_numAssign << std::endl;
            }
            if(verbose) {
                std::cerr << "counter=" << counter << std::endl;
            }

            auction_eps *= auction_factor;
        }
        cudaThreadSynchronize();

        // Stop timer
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        if(verbose) {
            std::cerr <<
                "run_num="         << run_num      <<
                " | h_numAssign="  << h_numAssign  <<
                " | milliseconds=" << milliseconds << std::endl;
        }

        cudaThreadSynchronize();
     }

    // Read out results
    cudaMemcpy(h_person2item, d_person2item, sizeof(int) * num_nodes, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_columns);
    cudaFree(d_offsets);
    cudaFree(d_person2item);
    cudaFree(d_item2person);
    cudaFree(d_bids);
    cudaFree(d_prices);
    cudaFree(d_bidders);
    cudaFree(d_sbids);
    cudaFree(d_numAssign);
    cudaFree(d_rand);

    return 0;
} // end run_auction

int run_auction_python(
    int    num_nodes,
    int    num_edges,

    float* h_data,      // data
    int*   h_offsets,   // offsets for items
    int*   h_columns,

    int*   h_person2item, // results

    float auction_max_eps,
    float auction_min_eps,
    float auction_factor,

    int num_runs,
    int verbose
) {

    cudaEvent_t auction_start, auction_stop;
    float milliseconds = 0;
    cudaEventCreate(&auction_start);
    cudaEventCreate(&auction_stop);
    cudaEventRecord(auction_start, 0);

    run_auction(
        num_nodes,
        num_edges,

        h_data,      // data
        h_offsets,   // offsets for items
        h_columns,

        h_person2item, // results

        auction_max_eps,
        auction_min_eps,
        auction_factor,

        num_runs,
        0
    );
    cudaEventRecord(auction_stop, 0);
    cudaEventSynchronize(auction_stop);
    cudaEventElapsedTime(&milliseconds, auction_start, auction_stop);
    cudaEventDestroy(auction_start);
    cudaEventDestroy(auction_stop);
    if(verbose > 0) {
        std::cerr << "run_auction     " << milliseconds << std::endl;
    }
    return 0;
}

int dot_auction(
        int num_nodes,
        int *Ap, int *Aj, double *Ax,
        int *Bp, int *Bj, double *Bx,
        int k,
        int *h_person2item,
        int verbose
) {

    std::chrono::high_resolution_clock::time_point topdot_start = std::chrono::high_resolution_clock::now();

    int* h_columns   = (int *)malloc(sizeof(int) * num_nodes * k);
    double* h_data_d = (double *)malloc(sizeof(double) * num_nodes * k);
    float* h_data    = (float *)malloc(sizeof(float) * num_nodes * k);
    int* h_offsets   = (int *)malloc(sizeof(int) * num_nodes + 1);
    _topdot(num_nodes, num_nodes, Ap, Aj, Ax, Bp, Bj, Bx, k, -1, h_columns, h_data_d);
    h_offsets[0] = 0;
    for(int i = 1; i < num_nodes + 1; i++) {
        h_offsets[i] = i * k;
    }

    for(int i = 0; i < num_nodes * k; i++) {
        h_data[i] = (float)h_data_d[i];
        if(verbose > 1) {
            std::cerr << h_columns[i] << ":" << h_data[i] << " ";
            if((i + 1) % k == 0) {
                std::cerr << std::endl;
            }
        }
    }
    free(h_data_d);

        // Stop timer
    std::chrono::high_resolution_clock::time_point topdot_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(topdot_stop - topdot_start);

    if(verbose > 0) {
        std::cerr << "topdot          " << 1000 * time_span.count() << std::endl;
    }

    // --
    // Auction algorithm

    cudaEvent_t auction_start, auction_stop;
    float milliseconds = 0;
    cudaEventCreate(&auction_start);
    cudaEventCreate(&auction_stop);
    cudaEventRecord(auction_start, 0);

    run_auction(
        (int)num_nodes,
        (int)num_nodes * k,

        h_data,
        h_offsets,
        h_columns,

        h_person2item,

        (float)1.0,
        (float)1.0,
        (float)0.0,

        (int)1, // 1 run
        (int)0  // not verbose
    );

    cudaEventRecord(auction_stop, 0);
    cudaEventSynchronize(auction_stop);
    cudaEventElapsedTime(&milliseconds, auction_start, auction_stop);
    cudaEventDestroy(auction_start);
    cudaEventDestroy(auction_stop);
    if(verbose > 0) {
        std::cerr << "run_auction     " << milliseconds << std::endl;
    }

    free(h_columns);
    free(h_data);
    free(h_offsets);

    return 0;
} // end dot_auction

} // end extern


void init_device() {
    // Init devices
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if(deviceCount<1){
        printf("There is no device detected.\n");
        exit(1);
    }
    int device=0;
    cudaDeviceProp deviceProp;
    for (device = 0; device < deviceCount; ++device) {
        if(cudaGetDeviceProperties(&deviceProp, device) == cudaSuccess) {
            if(deviceProp.major >= 1) {
                break;
            }
        }
    }
    if(device == deviceCount) {
        printf("There is no device supporting CUDA.\n");
        exit(1);
    }
    cudaSetDevice(device);
}


int main(int argc, char **argv)
{
#ifdef DENSE
    std::cerr << "auction_kernel_dense.cu" << std::endl;
#else
    std::cerr << "auction_kernel_csr.cu" << std::endl;
#endif

    init_device();

    // Load data
    float* raw_data = (float *)malloc(sizeof(float) * MAX_NODES * MAX_NODES);
    int num_nodes = load_data(raw_data);
    int num_edges = num_nodes * num_nodes;
    if(num_nodes <= 0) {
        return 1;
    }

    float* h_data  = (float *)realloc(raw_data, sizeof(float) * num_nodes * num_nodes);

    // Dense
    int* h_offsets = (int *)malloc(sizeof(int) * num_nodes + 1);
    h_offsets[0] = 0;
    for(int i = 1; i < num_nodes + 1; i++) {
        h_offsets[i] = i * num_nodes;
    }

    int* h_columns = (int *)malloc(sizeof(int) * num_edges);
    for(int i = 0; i < num_edges; i++) {
        h_columns[i] = i % num_nodes;
    }

    int* h_person2item = (int *)malloc(sizeof(int) * num_nodes);

    int verbose = 1;

    run_auction(
        num_nodes,
        num_edges,

        h_data,
        h_offsets,
        h_columns,

        h_person2item,

        AUCTION_MAX_EPS,
        AUCTION_MIN_EPS,
        AUCTION_FACTOR,

        NUM_RUNS,
        verbose
    );

    // Print results
    float score = 0;
    for (int i = 0; i < num_nodes; i++) {
        std::cout << i << " " << h_person2item[i] << std::endl;
        // score += h_data[i + num_nodes * h_person2item[i]];
        score += h_data[i * num_nodes + h_person2item[i]];
    }

    std::cerr << "score=" << (int)score << std::endl;

    free(h_data);
    free(h_offsets);
    free(h_columns);
    free(h_person2item);
}

#endif
