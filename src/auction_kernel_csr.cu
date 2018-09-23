#ifndef __AUCTION_VARS
// #define DETERMINISTIC
#endif

__global__ void run_bidding(
    const int num_nodes,
    
    float *data,
    int *offsets,
    int *columns,
    
    int *person2item,
    float *bids,
    int *bidders,
    int *sbids,
    float *prices,
    float auction_eps
)
{
    
    int i = blockDim.x * blockIdx.x + threadIdx.x; // person index
    if(i < num_nodes){
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
#ifdef DETERMINISTIC
            bids[i * num_nodes + top1_col] = bid;
            atomicMax(sbids + top1_col, 1);
#else
            int idx = atomicAdd(sbids + top1_col, 1);
            bids[num_nodes * top1_col + idx] = bid;
            bidders[num_nodes * top1_col + idx] = i;
#endif
        }
    }
}


__global__ void run_assignment(
    const int num_nodes,
    int *person2item,
    int *item2person,
    float *bids,
    int *bidders,
    int *sbids,
    float *prices,
    int *num_assigned
)
{
    
    int j = blockDim.x * blockIdx.x + threadIdx.x; // item index
    if(j < num_nodes) {
#ifdef DETERMINISTIC
        if(sbids[j] != 0) {
            float high_bid  = 0.0;
            int high_bidder = -1;
            
            float tmp_bid = -1;
            for(int i = 0; i < num_nodes; i++){        
                tmp_bid = bids[i * num_nodes + j]; 
                if(tmp_bid > high_bid){
                    high_bid    = tmp_bid;
                    high_bidder = i;
                }
            }
#else
        int num_bidders = sbids[j];
        if(num_bidders != 0) {
            float high_bid  = bids[num_nodes * j];
            int high_bidder = bidders[num_nodes * j];
            
            float tmp_bid = -1.0;
            for(int idx = 1; idx < num_bidders; idx++){
                tmp_bid = bids[num_nodes * j + idx];
                if(tmp_bid > high_bid){
                    high_bid    = tmp_bid;
                    high_bidder = bidders[num_nodes * j + idx];
                }
            }
#endif
            int current_person = item2person[j];
            if(current_person >= 0){
                person2item[current_person] = -1; 
            } else {
                atomicAdd(num_assigned, 1);                
            }
            
            prices[j]                += high_bid;
            person2item[high_bidder] = j;
            item2person[j]           = high_bidder;
        }
    }
}
