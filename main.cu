#ifndef MAIN_AUCTION
#define MAIN_AUCTION

#include <iostream>
#include <string>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include "auction_kernel.cu"

// --
// Define constants

#ifndef NUM_NODES    
#define NUM_NODES 5180 // Dimension of problem
#endif

#ifndef BLOCKSIZE
#define BLOCKSIZE 32 // How best to set this?
#endif

#ifndef AUCTION_EPS
#define AUCTION_EPS 1 // Larger values mean solution is more approximate
#endif

void load_data(int *h_data, const int dim) {
    std::ifstream input_file("graph", std::ios_base::in);
    
    std::cerr << "load_data: start" << std::endl;
    int i = 0;
    int val;
    while(input_file >> val) {
        h_data[i] = val;
        i++;
        if(i > NUM_NODES * NUM_NODES) {
            std::cerr << "load_data: ERROR -- data file too large" << std::endl;
            return;
        }
    }
    std::cerr << "load_data: finish" << std::endl;
}

int run_auction(){
    
    // Load data
    int* h_data = (int *)malloc(sizeof(int) * NUM_NODES * NUM_NODES);
    load_data(h_data, NUM_NODES);
    
    // Output data structure
    int* h_person2item = (int *)malloc(sizeof(int) * NUM_NODES);
    
    int* d_data;         //a [i,j] : desire of person i for object j
    int* d_bids;         //bids value
    int* d_sbids;
    int* d_prices;       //p[j] : each object j has a price:
    int* d_person2item;  //each person is or not assigned
    int* d_item2person;  //each object is or not assigned
    
    //using atomic operations, counts the number of assigns, 
    //otherwise, used as a boolean that is set whenever there is an unassigned person
    int* d_numAssign = 0;

    cudaMalloc((void **)&d_data,        sizeof(int)   * NUM_NODES * NUM_NODES);
    cudaMalloc((void **)&d_bids,        sizeof(int)   * NUM_NODES * NUM_NODES);
    cudaMalloc((void **)&d_sbids,       sizeof(int)   * NUM_NODES);
    cudaMalloc((void **)&d_prices,      sizeof(int)   * NUM_NODES);
    cudaMalloc((void **)&d_person2item, sizeof(int  ) * NUM_NODES);
    cudaMalloc((void **)&d_item2person, sizeof(int  ) * NUM_NODES);
    cudaMalloc((void **)&d_numAssign,   sizeof(int  ) * 1) ;
    
    cudaMemcpy(d_data, h_data, sizeof(int) * NUM_NODES * NUM_NODES, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCKSIZE, 1, 1);
    int gx = ceil(NUM_NODES / (double) dimBlock.x);
    dim3 dimGrid(gx, 1, 1);
    
    for(int run_num = 0; run_num < 4; run_num++) {
        
        // Reset data structures
        int h_numAssign = 0;
        cudaMemset(d_bids,           0, NUM_NODES * NUM_NODES * sizeof(int));
        cudaMemset(d_sbids,          0, NUM_NODES * sizeof(int));
        cudaMemset(d_prices,       0.0, NUM_NODES * sizeof(int));
        cudaMemset(d_person2item,   -1, NUM_NODES * sizeof(int));
        cudaMemset(d_item2person,   -1, NUM_NODES * sizeof(int));
        cudaMemset(d_numAssign,      0, 1 * sizeof(int));
        cudaThreadSynchronize();

        // Start timer
        cudaEvent_t start, stop;
        float milliseconds = 0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
            
        int stop_cond = NUM_NODES;
        while(h_numAssign < stop_cond){
                      
            cudaMemset(d_bids,  0, NUM_NODES * NUM_NODES * sizeof(int));
            cudaMemset(d_sbids, 0, NUM_NODES * sizeof(int));
            cudaThreadSynchronize();
                        
            run_bidding<<<dimBlock, dimGrid>>>(
                NUM_NODES,
                d_data,
                d_person2item,
                d_bids,
                d_sbids,
                d_prices,
                AUCTION_EPS
            );
            run_assignment<<<dimBlock, dimGrid>>>(
                NUM_NODES,
                d_person2item,
                d_item2person,
                d_bids,
                d_sbids,
                d_prices,
                d_numAssign
            );
            cudaThreadSynchronize();
            
            cudaMemcpy(&h_numAssign, d_numAssign, sizeof(int) * 1, cudaMemcpyDeviceToHost);
        }
        // Stop timer
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        std::cerr << "h_numAssign=" << h_numAssign << " | milliseconds=" << milliseconds << std::endl;

     }
     
     // Read out results
    cudaMemcpy(h_person2item, d_person2item, sizeof(int) * NUM_NODES, cudaMemcpyDeviceToHost);
    int score = 0;
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