#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>

void logic_gate_net(long long const *inp, long long *out) {
	const long long v0 = ~inp[4] | inp[5];
	const long long v1 = inp[4];
	const long long v2 = ~inp[7] | inp[6];
	const long long v3 = ~inp[0];
	const long long v4 = ~inp[3] | inp[2];
	const long long v5 = inp[1];
	out[0] = v1;
	out[1] = v1;
	out[2] = ~(v0 ^ v2);
	out[3] = v4 & ~v3;
	out[4] = v5;
	out[5] = ~v2;
}

void apply_logic_gate_net (bool const *inp, int *out, size_t len) {
    long long *inp_temp = malloc(8*sizeof(long long));
    long long *out_temp = malloc(6*sizeof(long long));
    long long *out_temp_o = malloc(2*sizeof(long long));
    
    for(size_t i = 0; i < len; ++i) {
    
        // Converting the bool array into a bitpacked array
        for(size_t d = 0; d < 8; ++d) {
            long long res = 0LL;
            for(size_t b = 0; b < 64; ++b) {
                res <<= 1;
                res += !!(inp[i * 8 * 64 + (64 - b - 1) * 8 + d]);
            }
            inp_temp[d] = res;
        }
    
        // Applying the logic gate net
        logic_gate_net(inp_temp, out_temp);
        
        // GroupSum of the results via logic gate networks
        for(size_t c = 0; c < 3; ++c) {  // for each class
            // Initialize the output bits
            for(size_t d = 0; d < 2; ++d) {
                out_temp_o[d] = 0LL;
            }
            
            // Apply the adder logic gate network
            for(size_t a = 0; a < 2; ++a) {
                long long carry = out_temp[c * 2 + a];
                long long out_temp_o_d;
                for(int d = 2 - 1; d >= 0; --d) {
                    out_temp_o_d  = out_temp_o[d];
                    out_temp_o[d] = carry ^ out_temp_o_d;
                    carry         = carry & out_temp_o_d;
                }
            }
            
            // Unpack the result bits
            for(size_t b = 0; b < 64; ++b) {
                const long long bit_mask = 1LL << b;
                int res = 0;
                for(size_t d = 0; d < 2; ++d) {
                    res <<= 1;
                    res += !!(out_temp_o[d] & bit_mask);
                }
                out[(i * 64 + b) * 3 + c] = res;
            }
        }
    }
    free(inp_temp);
    free(out_temp);
    free(out_temp_o);
}
