//
//  blake2b.metal
//  MetalTest
//
//  Created by A on 10.01.22.
//

#include <metal_stdlib>
using namespace metal;

#define SOURCE_SIZE 64
#define DIGEST_SIZE 32

constant static const uchar blake2b_sigma[12 * 16] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3,
    11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4,
    7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8,
    9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13,
    2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9,
    12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11,
    13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10,
    6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5,
    10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3,
};

enum Blake2b_IV
{
    iv0 = 0x6a09e667f3bcc908ul,
    iv1 = 0xbb67ae8584caa73bul,
    iv2 = 0x3c6ef372fe94f82bul,
    iv3 = 0xa54ff53a5f1d36f1ul,
    iv4 = 0x510e527fade682d1ul,
    iv5 = 0x9b05688c2b3e6c1ful,
    iv6 = 0x1f83d9abfb41bd6bul,
    iv7 = 0x5be0cd19137e2179ul,
};

ulong rotr64(ulong a, ulong shift) { return rotate(a, 64 - shift); }

#define G(r, i, a, b, c, d)                                                    \
    do {                                                                       \
        a = a + b + m[blake2b_sigma[r * 16 + 2 * i + 0]];                      \
        d = rotr64(d ^ a, 32);                                                 \
        c = c + d;                                                             \
        b = rotr64(b ^ c, 24);                                                 \
        a = a + b + m[blake2b_sigma[r * 16 + 2 * i + 1]];                      \
        d = rotr64(d ^ a, 16);                                                 \
        c = c + d;                                                             \
        b = rotr64(b ^ c, 63);                                                 \
    } while (0)

#define ROUND(r)                                                               \
    do {                                                                       \
        G(r, 0, v[0], v[4], v[8], v[12]);                                      \
        G(r, 1, v[1], v[5], v[9], v[13]);                                      \
        G(r, 2, v[2], v[6], v[10], v[14]);                                     \
        G(r, 3, v[3], v[7], v[11], v[15]);                                     \
        G(r, 4, v[0], v[5], v[10], v[15]);                                     \
        G(r, 5, v[1], v[6], v[11], v[12]);                                     \
        G(r, 6, v[2], v[7], v[8], v[13]);                                      \
        G(r, 7, v[3], v[4], v[9], v[14]);                                      \
    } while (0)

#define BLAKE2B_ROUNDS() ROUND(0);ROUND(1);ROUND(2);ROUND(3);ROUND(4);ROUND(5);ROUND(6);ROUND(7);ROUND(8);ROUND(9);ROUND(10);ROUND(11);

static void blake2b_256(thread ulong *h, thread const ulong* m)
{
    ulong v[16] =
    {
        iv0 ^ 0x01010020, iv1, iv2, iv3, iv4               , iv5,  iv6, iv7,
        iv0             , iv1, iv2, iv3, iv4 ^ SOURCE_SIZE , iv5, ~iv6, iv7,
    };

    BLAKE2B_ROUNDS();

    h[0] = v[0] ^ v[ 8] ^ iv0 ^ 0x01010020;
    h[1] = v[1] ^ v[ 9] ^ iv1;
    h[2] = v[2] ^ v[10] ^ iv2;
    h[3] = v[3] ^ v[11] ^ iv3;
}


kernel void merkle(device uchar* data, constant uint& round, constant uint& leaves, uint id [[thread_position_in_grid]])
{
   
    device  ulong* lhs = (device  ulong*)&data[DIGEST_SIZE * (id << (round + 1))];
    device  ulong* rhs = (device  ulong*)&data[DIGEST_SIZE * ((id << (round + 1)) + (1 << round))];
    
    ulong m[16] = {
        lhs[0], lhs[1], lhs[2], lhs[3], rhs[0], rhs[1], rhs[2], rhs[3], 0, 0, 0, 0, 0, 0, 0, 0
    };
    
    ulong hash[4];
    ::blake2b_256(hash, m);

    lhs[0] = hash[0];
    lhs[1] = hash[1];
    lhs[2] = hash[2];
    lhs[3] = hash[3];
}
