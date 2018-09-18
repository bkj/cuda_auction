#ifndef MAIN_AUCTION
#define MAIN_AUCTION

#include <iostream>
#include <string>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
// #include "nodet_auction_kernel.cu" // Faster, but non deterministic
#include "auction_kernel.cu"

// --
// Define constants

#ifndef MAX_NODES    
#define MAX_NODES 20000 // Dimension of problem
#endif

#ifndef BLOCKSIZE
#define BLOCKSIZE 32 // How best to set this?
#endif

#ifndef AUCTION_MAX_EPS
#define AUCTION_MAX_EPS 1.0 // Larger values mean solution is more approximate
#endif

#ifndef AUCTION_MIN_EPS
#define AUCTION_MIN_EPS 1.0 / 4.0
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

int run_auction(){
    
    // Load data
    float* raw_data = (float *)malloc(sizeof(float) * MAX_NODES * MAX_NODES);
    int NUM_NODES = load_data(raw_data);
    if(NUM_NODES <= 0) {
        return 1;
    }
    float* h_data = (float *)realloc(raw_data, sizeof(float) * NUM_NODES * NUM_NODES);
    
    int h_numAssign;
    int* h_person2item = (int *)malloc(sizeof(int) * NUM_NODES);
    
    float* d_data;
    float* d_bids;
    float* d_prices;
    int* d_bidders;
    int* d_sbids;
    int* d_person2item;
    int* d_item2person;
    
    //using atomic operations, counts the number of assigns, 
    //otherwise, used as a boolean that is set whenever there is an unassigned person
    int* d_numAssign = 0;

    cudaMalloc((void **)&d_data,        NUM_NODES * NUM_NODES * sizeof(float));
    cudaMalloc((void **)&d_bids,        NUM_NODES * NUM_NODES * sizeof(float));
    cudaMalloc((void **)&d_prices,      NUM_NODES             * sizeof(float));
    cudaMalloc((void **)&d_bidders,     NUM_NODES * NUM_NODES * sizeof(int));
    cudaMalloc((void **)&d_sbids,       NUM_NODES * sizeof(int));
    cudaMalloc((void **)&d_person2item, NUM_NODES * sizeof(int));
    cudaMalloc((void **)&d_item2person, NUM_NODES * sizeof(int));
    cudaMalloc((void **)&d_numAssign,           1 * sizeof(int)) ;
    
    cudaMemcpy(d_data, h_data, sizeof(float) * NUM_NODES * NUM_NODES, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCKSIZE, 1, 1);
    int gx = ceil(NUM_NODES / (double) dimBlock.x);
    dim3 dimGrid(gx, 1, 1);
    
    for(int run_num = 0; run_num < 4; run_num++) {
        
        // Start timer
        cudaEvent_t start, stop;
        float milliseconds = 0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        // Reset data structures
        cudaMemset(d_prices, 0.0, NUM_NODES * sizeof(float));
        cudaThreadSynchronize();

        float auction_eps = AUCTION_MAX_EPS;
        while(auction_eps >= AUCTION_MIN_EPS) {
            h_numAssign = 0;
            cudaMemset(d_bidders,        0, NUM_NODES * NUM_NODES * sizeof(int));
            cudaMemset(d_person2item,   -1, NUM_NODES             * sizeof(int));
            cudaMemset(d_item2person,   -1, NUM_NODES             * sizeof(int));
            cudaMemset(d_numAssign,      0, 1                     * sizeof(int));
            cudaThreadSynchronize();
            
            while(h_numAssign < NUM_NODES){
                          
                cudaMemset(d_bids,  0, NUM_NODES * NUM_NODES * sizeof(float));
                cudaMemset(d_sbids, 0, NUM_NODES             * sizeof(int));
                cudaThreadSynchronize();
                            
                run_bidding<<<dimBlock, dimGrid>>>(
                    NUM_NODES,
                    d_data,
                    d_person2item,
                    d_bids,
                    d_bidders,
                    d_sbids,
                    d_prices,
                    auction_eps
                );
                run_assignment<<<dimBlock, dimGrid>>>(
                    NUM_NODES,
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
            }
            auction_eps = auction_eps / 2.0;
        }
        
        // Stop timer
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        std::cerr << 
            "run_num="         << run_num      << 
            " | h_numAssign="  << h_numAssign  <<
            " | milliseconds=" << milliseconds << std::endl;
        
        cudaThreadSynchronize();
     }
     
     // Read out results
    cudaMemcpy(h_person2item, d_person2item, sizeof(int) * NUM_NODES, cudaMemcpyDeviceToHost);
    float score = 0;
     for (int i = 0; i < NUM_NODES; i++) {
         std::cout << i << " " << h_person2item[i] << std::endl;
         score += h_data[i + NUM_NODES * h_person2item[i]];
     }
     std::cerr << "score=" << score << std::endl;

     // Free memory
    cudaFree(d_data);
    cudaFree(d_bids);
    cudaFree(d_prices);  
    cudaFree(d_person2item); 
    cudaFree(d_item2person); 
    cudaFree(d_numAssign);
    
    free(h_data);
    free(h_person2item);
    
    return 0;
}


int main(int argc, char **argv)
{
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
    run_auction();
}

#endif