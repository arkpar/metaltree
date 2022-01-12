#include <metal_stdlib>
using namespace metal;

typedef ulong4 HashType;

inline bool cmp(ulong4 a, ulong4 b) {
    return a.x == b.x ? (a.y == b.y ? (a.z == b.z ? (a.w > b.w) : a.z > b.z) : a.y > b.y ) : a.x > b.x;
}

kernel void sort(device HashType *input [[buffer(0)]],
                            constant int &p [[buffer(1)]],
                            constant int &q [[buffer(2)]],
                            uint gid [[thread_position_in_grid]])
{
    int distance = 1 << (p-q);
    bool direction = ((gid >> p) & 2) == 0;
    

    if ((gid & distance) == 0 && (cmp(input[gid << 1] , input[(gid | distance) << 1])) == direction) {
        HashType temp0 = input[gid << 1];
        HashType temp1 = input[(gid << 1) + 1];
        input[gid << 1 ] = input[(gid | distance) << 1];
        input[(gid << 1) + 1 ] = input[((gid | distance) << 1) + 1];
        input[(gid | distance) << 1] = temp0;
        input[((gid | distance) << 1) + 1] = temp1;
    }
}
