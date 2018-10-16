#ifndef __AUCTION_HEADER
#define __AUCTION_HEADER

typedef struct {
    int row;
    int idx;
    float best_val;
    float next_best_val;
    float tiebreaker;
    bool is_first;
} Entry;

#endif