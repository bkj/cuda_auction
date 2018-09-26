/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <omp.h>

#ifndef __TOPDOT_VARS
#define EMPTY_COL -99
#endif

struct candidate {
    int index;
    double value;
};

bool candidate_cmp(candidate c_i, candidate c_j) {
    return (c_i.value > c_j.value);
}

void _topdot(
        int n_row,
        int n_col,
        int *Ap, int *Aj, double *Ax,
        int *Bp, int *Bj, double *Bx,
        int k,
        double lower_bound,
        int *Cj, double *Cx
)
{

    #pragma omp parallel for
    for(int row = 0; row < n_row; row++){

        std::vector<int>    next(n_col, -1);
        std::vector<double> sums(n_col,  0);
        std::vector<candidate> candidates;

        int head   = -2;
        int length =  0;

        // SpMM
        int a_offset_start = Ap[row];
        int a_offset_end   = Ap[row+1];

        for(int a_col_idx = a_offset_start; a_col_idx < a_offset_end; a_col_idx++){
            int a_col    = Aj[a_col_idx];
            double a_val = Ax[a_col_idx];

            int b_offset_start = Bp[a_col];
            int b_offset_end   = Bp[a_col + 1];
            for(int b_col_idx = b_offset_start; b_col_idx < b_offset_end; b_col_idx++){
                int b_col    = Bj[b_col_idx];
                double b_val = Bx[b_col_idx];
                sums[b_col] += a_val * b_val;

                // keep pointer to previous nonzero entry
                if(next[b_col] == -1){
                    next[b_col] = head;
                    head        = b_col;
                    length++;
                }
            }
        }

        // Collect results
        int num_candidates = 0;
        for(int i = 0; i < length; i++){
            if(sums[head] > lower_bound){
                candidate c;
                c.index = head;
                c.value = sums[head];
                candidates.push_back(c);
                num_candidates++;
            }

            head = next[head];
        }

        // Get top-k
        // >>
        // This makes a big difference! Whether ordered randomly
        // or sorted.  Why? Memory collisions?
        if (num_candidates > k){
            std::nth_element(
                candidates.begin(),
                candidates.begin() + k,
                candidates.end(),
                candidate_cmp
            );
        }
        // --
        // if (num_candidates > k){
        //     std::partial_sort(
        //         candidates.begin(),
        //         candidates.begin() + k,
        //         candidates.end(),
        //         candidate_cmp
        //     );
        // } else {
        //     std::sort(
        //         candidates.begin(),
        //         candidates.end(),
        //         candidate_cmp
        //     );
        // }
        // <<

        for(int entry_idx = 0; entry_idx < k; entry_idx++){
            if(entry_idx < num_candidates) {
                Cj[row * k + entry_idx] = candidates[entry_idx].index;
                Cx[row * k + entry_idx] = candidates[entry_idx].value;
            } else {
                Cj[row * k + entry_idx] = EMPTY_COL;
                Cx[row * k + entry_idx] = 0;
            }
        }
    }


    return;
}
