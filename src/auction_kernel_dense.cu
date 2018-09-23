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

            int fir_maxObj      = -1;
            float fir_maxObjValue = 0;
            float sec_maxObjValue = -1000;
            float temp_ObjValue   = 0;

            fir_maxObj      = 0;
            fir_maxObjValue = data[i * num_nodes] - prices[0];
            
            for(int j = 1; j < num_nodes; j++){
                temp_ObjValue = data[i * num_nodes + j] - prices[j];
                if(temp_ObjValue > fir_maxObjValue){
                    sec_maxObjValue = fir_maxObjValue;
                    
                    fir_maxObj      = j;
                    fir_maxObjValue = temp_ObjValue;
                } else if(temp_ObjValue > sec_maxObjValue){
                    sec_maxObjValue = temp_ObjValue;
                }        
            }

            float bid = fir_maxObjValue - sec_maxObjValue + auction_eps;
            bids[i * num_nodes + fir_maxObj] = bid;
            atomicMax(sbids + fir_maxObj, bid);
        }
    }
}


__global__ void run_assignment(
    const int num_nodes, int *person2item, int *item2person, float *bids, int *bidders, int *sbids, float *prices, int *num_assigned){
    
    int j = blockDim.x * blockIdx.x + threadIdx.x; // item index
    if(j < num_nodes) {
        if(sbids[j] != 0) {
            float high_bid    = 0.0;
            int high_bidder = -1;
            
            float tmp_bid = -1;
            for(int i = 0; i < num_nodes; i++){        
                tmp_bid = bids[i * num_nodes + j]; 
                if(tmp_bid > high_bid){
                    high_bid    = tmp_bid;
                    high_bidder = i;
                }
            }
            
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