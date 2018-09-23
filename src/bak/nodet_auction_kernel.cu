__global__ void run_bidding(
    const int num_nodes, float *data, int *person2item, float *bids, int *bidders, int *sbids, float *prices, float auction_eps){
    
    int i = blockDim.x * blockIdx.x + threadIdx.x; // person index
    if(i < num_nodes){
        if(person2item[i] == -1) {
            
            int fir_maxObj        = -1;
            float fir_maxObjValue = 0;
            float sec_maxObjValue = -1000;
            float temp_ObjValue   = 0;
            
            fir_maxObj      = 0;
            fir_maxObjValue = data[i] - prices[0];
            
            for(int j = 1; j < num_nodes; j++){
                temp_ObjValue = data[i + num_nodes * j] - prices[j];
                if(temp_ObjValue > fir_maxObjValue){
                    sec_maxObjValue = fir_maxObjValue;
                    
                    fir_maxObj      = j;
                    fir_maxObjValue = temp_ObjValue;
                } else if(temp_ObjValue > sec_maxObjValue){
                    sec_maxObjValue = temp_ObjValue;
                }        
            }
            
            float bid = data[i + num_nodes * fir_maxObj] - sec_maxObjValue + auction_eps;
            int idx = atomicAdd(sbids + fir_maxObj, 1);
            bids[idx + num_nodes * fir_maxObj] = bid;
            bidders[idx + num_nodes * fir_maxObj] = i;
        }
    }
}


__global__ void run_assignment(
    const int num_nodes, int *person2item, int *item2person, float *bids, int *bidders, int *sbids, float *prices, int *num_assigned){
    
    int j = blockDim.x * blockIdx.x + threadIdx.x; // item index
    if(j < num_nodes) {
        int num_bidders = sbids[j];
        if(num_bidders != 0) {
            float high_bid  = bids[0 + num_nodes * j];
            int high_bidder = bidders[0 + num_nodes * j];
            
            float tmp_bid = -1.0;
            for(int i = 1; i < num_bidders; i++){
                tmp_bid = bids[i + num_nodes * j];
                if(tmp_bid > high_bid){
                    high_bid    = tmp_bid;
                    high_bidder = bidders[i + num_nodes * j];
                }
            }
            
            int current_person = item2person[j];
            if(current_person >= 0){
                person2item[current_person] = -1; 
            } else {
                atomicAdd(num_assigned, 1);
            }
            
            prices[j]                = high_bid;
            person2item[high_bidder] = j;
            item2person[j]           = high_bidder;
        }
    }
}
