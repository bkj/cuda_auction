// auction.cu

#ifndef MAIN_AUCTION
#define MAIN_AUCTION

#include <iostream>
#include <string>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <curand.h>
#include <curand_kernel.h>

#include "auction.h"
#include <cub/cub.cuh>

// --
// Define constants


#ifndef __RUN_VARS
#define __RUN_VARS
#define MAX_NODES       20000 // Dimension of problem
#define AUCTION_MAX_EPS 1.0 // Larger values mean solution is more approximate
#define AUCTION_MIN_EPS 1.0
#define AUCTION_FACTOR  0.0
#define NUM_RUNS        10
#define BIG_NEGATIVE    -9999999

#define THREADS 1024

#endif

#include "topdot.cpp"
#include "auction_kernel_cub.cu"

// struct Min2Op {
//     __device__ __forceinline__
//     Entry operator()(const Entry &a, const Entry &b) const {
//         float best_val, next_best_val;
//         if(a.best_val < b.best_val) {
//             best_val = a.best_val;
//             next_best_val = min(a.next_best_val, b.best_val);
//         } else {
//             best_val = b.best_val;
//             next_best_val = min(b.next_best_val, a.best_val);
//         }
//         return (Entry){0, best_val, next_best_val, 0};
//     }
// };

struct BiddingOp {

    float* prices;
    CUB_RUNTIME_FUNCTION __forceinline__
    BiddingOp(float* prices) : prices(prices) {}

    __device__ __forceinline__
    Entry operator()(const Entry &a, const Entry &b) const {
        int best_row, best_idx;
        bool is_first;
        float best_val, next_best_val, tiebreaker;
        float a_val = a.best_val;
        float b_val = b.best_val;
        if(a.is_first) a_val -= prices[a.idx];
        if(b.is_first) b_val -= prices[b.idx];

        if(
            (a_val > b_val) ||
            ((a.best_val == b.best_val) && (a.idx < b.idx)) // Should (actually) break ties randomly
        ) {
            best_row      = a.row;
            best_idx      = a.idx;
            best_val      = a_val;
            next_best_val = max(a.next_best_val, b_val);
            tiebreaker    = a.tiebreaker;
            is_first      = false;
        } else {
            best_row      = b.row;
            best_idx      = b.idx;
            best_val      = b_val;
            next_best_val = max(a_val, b.next_best_val);
            tiebreaker    = b.tiebreaker;
            is_first      = false;
        }
        return (Entry){best_row, best_idx, best_val, next_best_val, tiebreaker, is_first};
    }
};

struct IsUnassigned
{
    __device__ __forceinline__
    bool operator()(const int &a) const {
        return a == -1;
    }
};



extern "C" {

__global__ void __make_entry_array(Entry* out, int* offsets, int* indices, float* data, float* rand_in, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x; // person index
    if(i < n) {
        int start = offsets[i];
        int end   = offsets[i + 1];
        for(int offset = start; offset < end; offset++) {
            out[offset] = (Entry){i, indices[offset], data[offset], BIG_NEGATIVE, rand_in[offset], true};
        }
    }
}

__global__ void __fill_price_array(Entry* out, float* in, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x; // person index
    if(i < n) {
        out[i] = (Entry){0, 0, in[i], 9999999, 0, false};
    }
}

__global__ void __setFlags(int* out, int* in, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x; // person index
    if(i < n) {
        out[i] = (int)(in[i] == -1);
    }
}

__global__ void __scatterBids(float* bids, int* sbids, Entry* in, int num_nodes, int n, float auction_eps) {
    int i = blockDim.x * blockIdx.x + threadIdx.x; // person index
    if(i < n) {
        float bid = in[i].best_val - in[i].next_best_val + auction_eps;
        bids[num_nodes * in[i].idx + in[i].row] = bid;
        atomicMax(sbids + in[i].idx, 1);
    }
}



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

    dim3 threadsPerBlock(512, 1, 1);
    dim3 blocksPerGrid(ceil(num_nodes / (double) threadsPerBlock.x), 1, 1);

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

    int* d_flags;
    cudaMalloc((void**)&d_flags, num_nodes * sizeof(int));

    int* d_num_unassigned;
    int* d_unassigned_offsets_start;
    int* d_unassigned_offsets_end;
    cudaMalloc((void**)&d_num_unassigned, 1 * sizeof(int));
    cudaMalloc((void**)&d_unassigned_offsets_start, num_nodes * sizeof(int));
    cudaMalloc((void**)&d_unassigned_offsets_end, num_nodes * sizeof(int));

    // --
    // Copy from host to device

    cudaMemcpy(d_data,    h_data,    num_edges       * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_columns, h_columns, num_edges       * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets, (num_nodes + 1) * sizeof(int),   cudaMemcpyHostToDevice);

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 123);
    curandGenerateUniform(gen, d_rand, num_nodes * num_nodes);

// >>
    Entry* d_entry_array;
    cudaMalloc((void**)&d_entry_array, num_edges * sizeof(Entry));
    int node_blocks = 1 + num_nodes / THREADS;
    __make_entry_array<<<node_blocks, THREADS>>>(d_entry_array, d_offsets, d_columns, d_data, d_rand, num_nodes);
// <<

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

            int counter = 0;
            while(h_numAssign < num_nodes){
                counter += 1;
                cudaMemset(d_bids,  0, num_nodes * num_nodes * sizeof(float));
                cudaMemset(d_sbids, 0, num_nodes * sizeof(int));

                // // ----------------------------------
                // // Find unassigned rows

                // void     *d_temp_storage = NULL;
                // size_t   temp_storage_bytes = 0;

                // __setFlags<<<node_blocks, THREADS>>>(d_flags, d_person2item, num_nodes);

                // cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_offsets, d_flags, d_unassigned_offsets_start,
                //     d_num_unassigned, num_nodes);
                // cudaMalloc(&d_temp_storage, temp_storage_bytes);
                // cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_offsets, d_flags, d_unassigned_offsets_start,
                //     d_num_unassigned, num_nodes);

                // cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_offsets + 1, d_flags, d_unassigned_offsets_end,
                //     d_num_unassigned, num_nodes);
                // cudaMalloc(&d_temp_storage, temp_storage_bytes);
                // cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_offsets + 1, d_flags, d_unassigned_offsets_end,
                //     d_num_unassigned, num_nodes);

                // int* h_num_unassigned = (int*)malloc(1 * sizeof(int));
                // cudaMemcpy(h_num_unassigned, d_num_unassigned, 1 * sizeof(int), cudaMemcpyDeviceToHost);
                // // std::cerr << "h_num_unassigned=" << h_num_unassigned[0] << std::endl;
                // // int* h_unassigned_offsets = (int*)malloc(h_num_selected_out[0] * sizeof(int));
                // // cudaMemcpy(h_unassigned_offsets, d_unassigned_offsets, h_num_selected_out[0] * sizeof(int), cudaMemcpyDeviceToHost);
                // // for(int i = 0; i < 10; i++)
                //     // std::cerr << h_unassigned_offsets[i] << std::endl;

                // // ----------------------------------
                // // Run bidding op on unassigned rows

                // d_temp_storage = NULL; temp_storage_bytes = 0;
                // BiddingOp bidding_op(d_prices);
                // Entry null_bid = {-1, -1, BIG_NEGATIVE, BIG_NEGATIVE, BIG_NEGATIVE};
                // Entry* d_entry_bid;
                // cudaMalloc((void**)&d_entry_bid, h_num_unassigned[0] * sizeof(Entry));
                // cub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes, d_entry_array, d_entry_bid,
                //     h_num_unassigned[0], d_unassigned_offsets_start, d_unassigned_offsets_end, bidding_op, null_bid);
                // cudaMalloc(&d_temp_storage, temp_storage_bytes);
                // cub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes, d_entry_array, d_entry_bid,
                //     h_num_unassigned[0], d_unassigned_offsets_start, d_unassigned_offsets_end, bidding_op, null_bid);

                // // Entry* h_entry_bid = (Entry*)malloc(h_num_unassigned[0] * sizeof(Entry));
                // // cudaMemcpy(h_entry_bid, d_entry_bid, h_num_unassigned[0] * sizeof(Entry), cudaMemcpyDeviceToHost);
                // // for(int i = 0; i < min(h_num_unassigned[0], 10); i++) {
                // //     Entry tmp = h_entry_bid[i];
                // //     std::cerr << tmp.row << " " << tmp.idx << " " << tmp.best_val << " " << tmp.next_best_val << std::endl;
                // // }

                // // ----------------------------------
                // // Broadcast bids to d_bids

                // int tmp_blocks = 1 + h_num_unassigned[0] / THREADS;
                // __scatterBids<<<tmp_blocks, THREADS>>>(d_bids, d_sbids, d_entry_bid, num_nodes, h_num_unassigned[0], auction_eps);

                // cudaMemset(d_bids,  0, num_nodes * num_nodes * sizeof(float));
                // cudaMemset(d_sbids, 0, num_nodes * sizeof(int));
                run_bidding<<<blocksPerGrid, threadsPerBlock>>>(
                    num_nodes,

                    d_data,
                    d_offsets,
                    d_columns,

                    // d_entry_array,

                    d_person2item,
                    d_bids,
                    d_bidders,
                    d_sbids,
                    d_prices,
                    auction_eps,
                    d_rand
                );

                run_assignment<<<blocksPerGrid, threadsPerBlock>>>(
                    num_nodes,
                    d_person2item,
                    d_item2person,
                    d_bids,
                    d_bidders,
                    d_sbids,
                    d_prices,
                    d_numAssign
                );

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

    // std::chrono::high_resolution_clock::time_point topdot_start = std::chrono::high_resolution_clock::now();

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
    init_device();

    std::cerr << "loading ./graph" << std::endl;
    std::vector<int> offsets;
    std::vector<int> columns;
    std::vector<float> data;

    std::ifstream input_file("graph", std::ios_base::in);
    int src, dst;
    float val;

    int last_src = -1;
    int i = 0;
    while(input_file >> src >> dst >> val) {
        if(src != last_src) {
            offsets.push_back(i);
            last_src = src;
        }
        columns.push_back(dst);
        data.push_back(val);
        i++;
    }
    offsets.push_back(i);

    int* h_offsets = &offsets[0];
    int* h_columns = &columns[0];
    float* h_data  = &data[0];

    int num_nodes = offsets.size() - 1;
    int num_edges = columns.size();
    std::cerr << "\t" << num_nodes << " " << num_edges << std::endl;

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
        // std::cout << i << " " << h_person2item[i] << std::endl;
        // score += h_data[i + num_nodes * h_person2item[i]];
        score += h_data[i * num_nodes + h_person2item[i]];
    }

    std::cerr << "score=" << (int)score << std::endl;

    offsets.clear();
    columns.clear();
    data.clear();
}

#endif