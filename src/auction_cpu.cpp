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
#include <string.h>
#include <omp.h>

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
    
    float* data,      // data
    int*   offsets,   // offsets for items
    int*   columns,
    
    int*   person2item, // results
    
    float auction_max_eps,
    float auction_min_eps,
    float auction_factor,
    
    int num_runs,
    int verbose
)
{
    
    // --
    // Declare variables
    
    int  num_assigned   = 0;
    int* item2person = (int*)malloc(num_nodes * sizeof(int));
    float* bids      = (float*)malloc(num_nodes * num_nodes * sizeof(float));
    float* prices    = (float*)malloc(num_nodes * sizeof(float));
    int* bidders     = (int*)malloc(num_nodes * num_nodes * sizeof(int)); // unused
    int* sbids       = (int*)malloc(num_nodes * sizeof(int));

    // --
    // Copy from host to device
        
    for(int run_num = 0; run_num < num_runs; run_num++) {
        
        for(int i = 0; i < num_nodes; i++) {
            prices[i] = 0.0;
            person2item[i] = -1;
        }
        
        // Start timer
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        
        float auction_eps = auction_max_eps;
        while(auction_eps >= auction_min_eps) {
            num_assigned = 0;
            
            for(int i = 0; i < num_nodes * num_nodes; i++) {
                bidders[i] = 0;
            }
            
            for(int i = 0; i < num_nodes; i++) {
                person2item[i] = -1;
                item2person[i] = -1;
            }
            num_assigned = 0;
            
            int counter = 0;
            while(num_assigned < num_nodes){
                counter += 1;

                memset(bids, 0, num_nodes * num_nodes * sizeof(float));
                memset(sbids, 0, num_nodes * sizeof(int));
                
                // #pragma omp parallel for num_threads(1)
                for(int i = 0; i < num_nodes; i++) {
                    if(person2item[i] == -1) {

                        int start_idx = offsets[i];
                        int end_idx   = offsets[i + 1];
                        
                        int top1_col   = columns[start_idx];
                        float top1_val = data[start_idx] - prices[top1_col];

                        float top2_val = -1000;
                        
                        int col;
                        float tmp_val;
                        for(int idx = start_idx; idx < end_idx; idx++){
                            col = columns[idx];
                            tmp_val = data[idx] - prices[col];
                            
                            if(tmp_val > top1_val){
                                top2_val = top1_val;
                                
                                top1_col = col;
                                top1_val = tmp_val;
                            } else if(tmp_val > top2_val){
                                top2_val = tmp_val;
                            }        
                        }
                        
                        float bid = top1_val - top2_val + auction_eps;
                        bids[num_nodes * top1_col + i] = bid;
                        sbids[top1_col] = 1;
                    }
                }
                
                // #pragma omp parallel for num_threads(1)
                for(int j = 0; j < num_nodes; j++) {
                    if(sbids[j] != 0) {
                        float high_bid  = 0.0;
                        int high_bidder = -1;
                        
                        float tmp_bid = -1;
                        for(int i = 0; i < num_nodes; i++){
                            tmp_bid = bids[num_nodes * j + i]; 
                            if(tmp_bid > high_bid){
                                high_bid    = tmp_bid;
                                high_bidder = i;
                            }
                        }
                        int current_person = item2person[j];
                        if(current_person >= 0){
                            person2item[current_person] = -1; 
                        } else {
                            // #pragma omp atomic
                            num_assigned++;
                        }
                        
                        prices[j]                += high_bid;
                        person2item[high_bidder] = j;
                        item2person[j]           = high_bidder;
                    }
                }
            }
            
            std::cerr << "counter=" << counter << std::endl;
            
            auction_eps *= auction_factor;
        }

        // Print results
        float score = 0;
        for (int i = 0; i < num_nodes; i++) {
            std::cout << i << " " << person2item[i] << std::endl;
            score += data[i * num_nodes + person2item[i]];
        }
        
        std::cerr << "score=" << (int)score << std::endl;  
        
        // Stop timer
        std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
        if(verbose) {
            std::cerr << 
                "run_num="         << run_num      << 
                " | num_assigned="  << num_assigned  <<
                " | milliseconds=" << 1000 * time_span.count() << std::endl;            
        }
     }
     
    
    free(bids);
    free(prices);  
    free(item2person); 
        
    return 0;
} // end run_auction

} // end extern



int main(int argc, char **argv)
{

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
    
    std::cerr << "num_nodes=" << num_nodes << std::endl;
    
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
