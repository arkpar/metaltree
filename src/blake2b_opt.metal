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

static void mix_scalar(device ulong* sh, thread ulong* m)
{
    ulong m00 = m[00];
    ulong m01 = m[01];
    ulong m02 = m[02];
    ulong m03 = m[03];
    ulong m04 = m[04];
    ulong m05 = m[05];
    ulong m06 = m[06];
    ulong m07 = m[07];

    ulong v00 = iv0 ^ 0x01010020;
    ulong v01 = iv1;
    ulong v02 = iv2;
    ulong v03 = iv3;
    ulong v04 = iv4;
    ulong v05 = iv5;
    ulong v06 = iv6;
    ulong v07 = iv7;

    ulong v08 = iv0;
    ulong v09 = iv1;
    ulong v10 = iv2;
    ulong v11 = iv3;
    ulong v12 = iv4 ^ SOURCE_SIZE;
    ulong v13 = iv5;
    ulong v14 = ~iv6;
    ulong v15 = iv7;

    //ROUND 1
    v00 += m00;
    v00 += v04;
    v12 ^= v00;
    v12 = (v12 >> 32) ^ (v12 << 32);
    v08 += v12;
    v04 ^= v08;
    v04 = (v04 >> 24) ^ (v04 << 40);

    v01 += m02;
    v01 += v05;
    v13 ^= v01;
    v13 = (v13 >> 32) ^ (v13 << 32);
    v09 += v13;
    v05 ^= v09;
    v05 = (v05 >> 24) ^ (v05 << 40);

    v02 += m04;
    v02 += v06;
    v14 ^= v02;
    v14 = (v14 >> 32) ^ (v14 << 32);
    v10 += v14;
    v06 ^= v10;
    v06 = (v06 >> 24) ^ (v06 << 40);

    v03 += m06;
    v03 += v07;
    v15 ^= v03;
    v15 = (v15 >> 32) ^ (v15 << 32);
    v11 += v15;
    v07 ^= v11;
    v07 = (v07 >> 24) ^ (v07 << 40);

    v02 += m05;
    v02 += v06;
    v14 ^= v02;
    v14 = (v14 >> 16) ^ (v14 << 48);
    v10 += v14;
    v06 ^= v10;
    v06 = (v06 >> 63) ^ (v06 <<  1);

    v03 += m07;
    v03 += v07;
    v15 ^= v03;
    v15 = (v15 >> 16) ^ (v15 << 48);
    v11 += v15;
    v07 ^= v11;
    v07 = (v07 >> 63) ^ (v07 <<  1);

    v00 += m01;
    v00 += v04;
    v12 ^= v00;
    v12 = (v12 >> 16) ^ (v12 << 48);
    v08 += v12;
    v04 ^= v08;
    v04 = (v04 >> 63) ^ (v04 <<  1);

    v01 += m03;
    v01 += v05;
    v13 ^= v01;
    v13 = (v13 >> 16) ^ (v13 << 48);
    v09 += v13;
    v05 ^= v09;
    v05 = (v05 >> 63) ^ (v05 <<  1);

    v00 += v05;
    v15 ^= v00;
    v15 = (v15 >> 32) ^ (v15 << 32);
    v10 += v15;
    v05 ^= v10;
    v05 = (v05 >> 24) ^ (v05 << 40);

    v01 += v06;
    v12 ^= v01;
    v12 = (v12 >> 32) ^ (v12 << 32);
    v11 += v12;
    v06 ^= v11;
    v06 = (v06 >> 24) ^ (v06 << 40);

    v02 += v07;
    v13 ^= v02;
    v13 = (v13 >> 32) ^ (v13 << 32);
    v08 += v13;
    v07 ^= v08;
    v07 = (v07 >> 24) ^ (v07 << 40);

    v03 += v04;
    v14 ^= v03;
    v14 = (v14 >> 32) ^ (v14 << 32);
    v09 += v14;
    v04 ^= v09;
    v04 = (v04 >> 24) ^ (v04 << 40);

    v02 += v07;
    v13 ^= v02;
    v13 = (v13 >> 16) ^ (v13 << 48);
    v08 += v13;
    v07 ^= v08;
    v07 = (v07 >> 63) ^ (v07 <<  1);

    v03 += v04;
    v14 ^= v03;
    v14 = (v14 >> 16) ^ (v14 << 48);
    v09 += v14;
    v04 ^= v09;
    v04 = (v04 >> 63) ^ (v04 <<  1);

    v00 += v05;
    v15 ^= v00;
    v15 = (v15 >> 16) ^ (v15 << 48);
    v10 += v15;
    v05 ^= v10;
    v05 = (v05 >> 63) ^ (v05 <<  1);

    v01 += v06;
    v12 ^= v01;
    v12 = (v12 >> 16) ^ (v12 << 48);
    v11 += v12;
    v06 ^= v11;
    v06 = (v06 >> 63) ^ (v06 <<  1);

    //ROUND 2
    v00 += v04;
    v12 ^= v00;
    v12 = (v12 >> 32) ^ (v12 << 32);
    v08 += v12;
    v04 ^= v08;
    v04 = (v04 >> 24) ^ (v04 << 40);

    v01 += m04;
    v01 += v05;
    v13 ^= v01;
    v13 = (v13 >> 32) ^ (v13 << 32);
    v09 += v13;
    v05 ^= v09;
    v05 = (v05 >> 24) ^ (v05 << 40);

    v02 += v06;
    v14 ^= v02;
    v14 = (v14 >> 32) ^ (v14 << 32);
    v10 += v14;
    v06 ^= v10;
    v06 = (v06 >> 24) ^ (v06 << 40);

    v03 += v07;
    v15 ^= v03;
    v15 = (v15 >> 32) ^ (v15 << 32);
    v11 += v15;
    v07 ^= v11;
    v07 = (v07 >> 24) ^ (v07 << 40);

    v02 += v06;
    v14 ^= v02;
    v14 = (v14 >> 16) ^ (v14 << 48);
    v10 += v14;
    v06 ^= v10;
    v06 = (v06 >> 63) ^ (v06 <<  1);

    v03 += m06;
    v03 += v07;
    v15 ^= v03;
    v15 = (v15 >> 16) ^ (v15 << 48);
    v11 += v15;
    v07 ^= v11;
    v07 = (v07 >> 63) ^ (v07 <<  1);

    v00 += v04;
    v12 ^= v00;
    v12 = (v12 >> 16) ^ (v12 << 48);
    v08 += v12;
    v04 ^= v08;
    v04 = (v04 >> 63) ^ (v04 <<  1);

    v01 += v05;
    v13 ^= v01;
    v13 = (v13 >> 16) ^ (v13 << 48);
    v09 += v13;
    v05 ^= v09;
    v05 = (v05 >> 63) ^ (v05 <<  1);

    v00 += m01;
    v00 += v05;
    v15 ^= v00;
    v15 = (v15 >> 32) ^ (v15 << 32);
    v10 += v15;
    v05 ^= v10;
    v05 = (v05 >> 24) ^ (v05 << 40);

    v01 += m00;
    v01 += v06;
    v12 ^= v01;
    v12 = (v12 >> 32) ^ (v12 << 32);
    v11 += v12;
    v06 ^= v11;
    v06 = (v06 >> 24) ^ (v06 << 40);

    v02 += v07;
    v13 ^= v02;
    v13 = (v13 >> 32) ^ (v13 << 32);
    v08 += v13;
    v07 ^= v08;
    v07 = (v07 >> 24) ^ (v07 << 40);

    v03 += m05;
    v03 += v04;
    v14 ^= v03;
    v14 = (v14 >> 32) ^ (v14 << 32);
    v09 += v14;
    v04 ^= v09;
    v04 = (v04 >> 24) ^ (v04 << 40);

    v02 += m07;
    v02 += v07;
    v13 ^= v02;
    v13 = (v13 >> 16) ^ (v13 << 48);
    v08 += v13;
    v07 ^= v08;
    v07 = (v07 >> 63) ^ (v07 <<  1);

    v03 += m03;
    v03 += v04;
    v14 ^= v03;
    v14 = (v14 >> 16) ^ (v14 << 48);
    v09 += v14;
    v04 ^= v09;
    v04 = (v04 >> 63) ^ (v04 <<  1);

    v00 += v05;
    v15 ^= v00;
    v15 = (v15 >> 16) ^ (v15 << 48);
    v10 += v15;
    v05 ^= v10;
    v05 = (v05 >> 63) ^ (v05 <<  1);

    v01 += m02;
    v01 += v06;
    v12 ^= v01;
    v12 = (v12 >> 16) ^ (v12 << 48);
    v11 += v12;
    v06 ^= v11;
    v06 = (v06 >> 63) ^ (v06 <<  1);

    //ROUND 3
    v00 += v04;
    v12 ^= v00;
    v12 = (v12 >> 32) ^ (v12 << 32);
    v08 += v12;
    v04 ^= v08;
    v04 = (v04 >> 24) ^ (v04 << 40);

    v01 += v05;
    v13 ^= v01;
    v13 = (v13 >> 32) ^ (v13 << 32);
    v09 += v13;
    v05 ^= v09;
    v05 = (v05 >> 24) ^ (v05 << 40);

    v02 += m05;
    v02 += v06;
    v14 ^= v02;
    v14 = (v14 >> 32) ^ (v14 << 32);
    v10 += v14;
    v06 ^= v10;
    v06 = (v06 >> 24) ^ (v06 << 40);

    v03 += v07;
    v15 ^= v03;
    v15 = (v15 >> 32) ^ (v15 << 32);
    v11 += v15;
    v07 ^= v11;
    v07 = (v07 >> 24) ^ (v07 << 40);

    v02 += m02;
    v02 += v06;
    v14 ^= v02;
    v14 = (v14 >> 16) ^ (v14 << 48);
    v10 += v14;
    v06 ^= v10;
    v06 = (v06 >> 63) ^ (v06 <<  1);

    v03 += v07;
    v15 ^= v03;
    v15 = (v15 >> 16) ^ (v15 << 48);
    v11 += v15;
    v07 ^= v11;
    v07 = (v07 >> 63) ^ (v07 <<  1);

    v00 += v04;
    v12 ^= v00;
    v12 = (v12 >> 16) ^ (v12 << 48);
    v08 += v12;
    v04 ^= v08;
    v04 = (v04 >> 63) ^ (v04 <<  1);

    v01 += m00;
    v01 += v05;
    v13 ^= v01;
    v13 = (v13 >> 16) ^ (v13 << 48);
    v09 += v13;
    v05 ^= v09;
    v05 = (v05 >> 63) ^ (v05 <<  1);

    v00 += v05;
    v15 ^= v00;
    v15 = (v15 >> 32) ^ (v15 << 32);
    v10 += v15;
    v05 ^= v10;
    v05 = (v05 >> 24) ^ (v05 << 40);

    v01 += m03;
    v01 += v06;
    v12 ^= v01;
    v12 = (v12 >> 32) ^ (v12 << 32);
    v11 += v12;
    v06 ^= v11;
    v06 = (v06 >> 24) ^ (v06 << 40);

    v02 += m07;
    v02 += v07;
    v13 ^= v02;
    v13 = (v13 >> 32) ^ (v13 << 32);
    v08 += v13;
    v07 ^= v08;
    v07 = (v07 >> 24) ^ (v07 << 40);

    v03 += v04;
    v14 ^= v03;
    v14 = (v14 >> 32) ^ (v14 << 32);
    v09 += v14;
    v04 ^= v09;
    v04 = (v04 >> 24) ^ (v04 << 40);

    v02 += m01;
    v02 += v07;
    v13 ^= v02;
    v13 = (v13 >> 16) ^ (v13 << 48);
    v08 += v13;
    v07 ^= v08;
    v07 = (v07 >> 63) ^ (v07 <<  1);

    v03 += m04;
    v03 += v04;
    v14 ^= v03;
    v14 = (v14 >> 16) ^ (v14 << 48);
    v09 += v14;
    v04 ^= v09;
    v04 = (v04 >> 63) ^ (v04 <<  1);

    v00 += v05;
    v15 ^= v00;
    v15 = (v15 >> 16) ^ (v15 << 48);
    v10 += v15;
    v05 ^= v10;
    v05 = (v05 >> 63) ^ (v05 <<  1);

    v01 += m06;
    v01 += v06;
    v12 ^= v01;
    v12 = (v12 >> 16) ^ (v12 << 48);
    v11 += v12;
    v06 ^= v11;
    v06 = (v06 >> 63) ^ (v06 <<  1);

    //ROUND 4
    v00 += m07;
    v00 += v04;
    v12 ^= v00;
    v12 = (v12 >> 32) ^ (v12 << 32);
    v08 += v12;
    v04 ^= v08;
    v04 = (v04 >> 24) ^ (v04 << 40);

    v01 += m03;
    v01 += v05;
    v13 ^= v01;
    v13 = (v13 >> 32) ^ (v13 << 32);
    v09 += v13;
    v05 ^= v09;
    v05 = (v05 >> 24) ^ (v05 << 40);

    v02 += v06;
    v14 ^= v02;
    v14 = (v14 >> 32) ^ (v14 << 32);
    v10 += v14;
    v06 ^= v10;
    v06 = (v06 >> 24) ^ (v06 << 40);

    v03 += v07;
    v15 ^= v03;
    v15 = (v15 >> 32) ^ (v15 << 32);
    v11 += v15;
    v07 ^= v11;
    v07 = (v07 >> 24) ^ (v07 << 40);

    v02 += v06;
    v14 ^= v02;
    v14 = (v14 >> 16) ^ (v14 << 48);
    v10 += v14;
    v06 ^= v10;
    v06 = (v06 >> 63) ^ (v06 <<  1);

    v03 += v07;
    v15 ^= v03;
    v15 = (v15 >> 16) ^ (v15 << 48);
    v11 += v15;
    v07 ^= v11;
    v07 = (v07 >> 63) ^ (v07 <<  1);

    v00 += v04;
    v12 ^= v00;
    v12 = (v12 >> 16) ^ (v12 << 48);
    v08 += v12;
    v04 ^= v08;
    v04 = (v04 >> 63) ^ (v04 <<  1);

    v01 += m01;
    v01 += v05;
    v13 ^= v01;
    v13 = (v13 >> 16) ^ (v13 << 48);
    v09 += v13;
    v05 ^= v09;
    v05 = (v05 >> 63) ^ (v05 <<  1);

    v00 += m02;
    v00 += v05;
    v15 ^= v00;
    v15 = (v15 >> 32) ^ (v15 << 32);
    v10 += v15;
    v05 ^= v10;
    v05 = (v05 >> 24) ^ (v05 << 40);

    v01 += m05;
    v01 += v06;
    v12 ^= v01;
    v12 = (v12 >> 32) ^ (v12 << 32);
    v11 += v12;
    v06 ^= v11;
    v06 = (v06 >> 24) ^ (v06 << 40);

    v02 += m04;
    v02 += v07;
    v13 ^= v02;
    v13 = (v13 >> 32) ^ (v13 << 32);
    v08 += v13;
    v07 ^= v08;
    v07 = (v07 >> 24) ^ (v07 << 40);

    v03 += v04;
    v14 ^= v03;
    v14 = (v14 >> 32) ^ (v14 << 32);
    v09 += v14;
    v04 ^= v09;
    v04 = (v04 >> 24) ^ (v04 << 40);

    v02 += m00;
    v02 += v07;
    v13 ^= v02;
    v13 = (v13 >> 16) ^ (v13 << 48);
    v08 += v13;
    v07 ^= v08;
    v07 = (v07 >> 63) ^ (v07 <<  1);

    v03 += v04;
    v14 ^= v03;
    v14 = (v14 >> 16) ^ (v14 << 48);
    v09 += v14;
    v04 ^= v09;
    v04 = (v04 >> 63) ^ (v04 <<  1);

    v00 += m06;
    v00 += v05;
    v15 ^= v00;
    v15 = (v15 >> 16) ^ (v15 << 48);
    v10 += v15;
    v05 ^= v10;
    v05 = (v05 >> 63) ^ (v05 <<  1);

    v01 += v06;
    v12 ^= v01;
    v12 = (v12 >> 16) ^ (v12 << 48);
    v11 += v12;
    v06 ^= v11;
    v06 = (v06 >> 63) ^ (v06 <<  1);

    //ROUND 5
    v00 += v04;
    v12 ^= v00;
    v12 = (v12 >> 32) ^ (v12 << 32);
    v08 += v12;
    v04 ^= v08;
    v04 = (v04 >> 24) ^ (v04 << 40);

    v01 += m05;
    v01 += v05;
    v13 ^= v01;
    v13 = (v13 >> 32) ^ (v13 << 32);
    v09 += v13;
    v05 ^= v09;
    v05 = (v05 >> 24) ^ (v05 << 40);

    v02 += m02;
    v02 += v06;
    v14 ^= v02;
    v14 = (v14 >> 32) ^ (v14 << 32);
    v10 += v14;
    v06 ^= v10;
    v06 = (v06 >> 24) ^ (v06 << 40);

    v03 += v07;
    v15 ^= v03;
    v15 = (v15 >> 32) ^ (v15 << 32);
    v11 += v15;
    v07 ^= v11;
    v07 = (v07 >> 24) ^ (v07 << 40);

    v02 += m04;
    v02 += v06;
    v14 ^= v02;
    v14 = (v14 >> 16) ^ (v14 << 48);
    v10 += v14;
    v06 ^= v10;
    v06 = (v06 >> 63) ^ (v06 <<  1);

    v03 += v07;
    v15 ^= v03;
    v15 = (v15 >> 16) ^ (v15 << 48);
    v11 += v15;
    v07 ^= v11;
    v07 = (v07 >> 63) ^ (v07 <<  1);

    v00 += m00;
    v00 += v04;
    v12 ^= v00;
    v12 = (v12 >> 16) ^ (v12 << 48);
    v08 += v12;
    v04 ^= v08;
    v04 = (v04 >> 63) ^ (v04 <<  1);

    v01 += m07;
    v01 += v05;
    v13 ^= v01;
    v13 = (v13 >> 16) ^ (v13 << 48);
    v09 += v13;
    v05 ^= v09;
    v05 = (v05 >> 63) ^ (v05 <<  1);

    v00 += v05;
    v15 ^= v00;
    v15 = (v15 >> 32) ^ (v15 << 32);
    v10 += v15;
    v05 ^= v10;
    v05 = (v05 >> 24) ^ (v05 << 40);

    v01 += v06;
    v12 ^= v01;
    v12 = (v12 >> 32) ^ (v12 << 32);
    v11 += v12;
    v06 ^= v11;
    v06 = (v06 >> 24) ^ (v06 << 40);

    v02 += m06;
    v02 += v07;
    v13 ^= v02;
    v13 = (v13 >> 32) ^ (v13 << 32);
    v08 += v13;
    v07 ^= v08;
    v07 = (v07 >> 24) ^ (v07 << 40);

    v03 += m03;
    v03 += v04;
    v14 ^= v03;
    v14 = (v14 >> 32) ^ (v14 << 32);
    v09 += v14;
    v04 ^= v09;
    v04 = (v04 >> 24) ^ (v04 << 40);

    v02 += v07;
    v13 ^= v02;
    v13 = (v13 >> 16) ^ (v13 << 48);
    v08 += v13;
    v07 ^= v08;
    v07 = (v07 >> 63) ^ (v07 <<  1);

    v03 += v04;
    v14 ^= v03;
    v14 = (v14 >> 16) ^ (v14 << 48);
    v09 += v14;
    v04 ^= v09;
    v04 = (v04 >> 63) ^ (v04 <<  1);

    v00 += m01;
    v00 += v05;
    v15 ^= v00;
    v15 = (v15 >> 16) ^ (v15 << 48);
    v10 += v15;
    v05 ^= v10;
    v05 = (v05 >> 63) ^ (v05 <<  1);

    v01 += v06;
    v12 ^= v01;
    v12 = (v12 >> 16) ^ (v12 << 48);
    v11 += v12;
    v06 ^= v11;
    v06 = (v06 >> 63) ^ (v06 <<  1);

    //ROUND 6
    v00 += m02;
    v00 += v04;
    v12 ^= v00;
    v12 = (v12 >> 32) ^ (v12 << 32);
    v08 += v12;
    v04 ^= v08;
    v04 = (v04 >> 24) ^ (v04 << 40);

    v01 += m06;
    v01 += v05;
    v13 ^= v01;
    v13 = (v13 >> 32) ^ (v13 << 32);
    v09 += v13;
    v05 ^= v09;
    v05 = (v05 >> 24) ^ (v05 << 40);

    v02 += m00;
    v02 += v06;
    v14 ^= v02;
    v14 = (v14 >> 32) ^ (v14 << 32);
    v10 += v14;
    v06 ^= v10;
    v06 = (v06 >> 24) ^ (v06 << 40);

    v03 += v07;
    v15 ^= v03;
    v15 = (v15 >> 32) ^ (v15 << 32);
    v11 += v15;
    v07 ^= v11;
    v07 = (v07 >> 24) ^ (v07 << 40);

    v02 += v06;
    v14 ^= v02;
    v14 = (v14 >> 16) ^ (v14 << 48);
    v10 += v14;
    v06 ^= v10;
    v06 = (v06 >> 63) ^ (v06 <<  1);

    v03 += m03;
    v03 += v07;
    v15 ^= v03;
    v15 = (v15 >> 16) ^ (v15 << 48);
    v11 += v15;
    v07 ^= v11;
    v07 = (v07 >> 63) ^ (v07 <<  1);

    v00 += v04;
    v12 ^= v00;
    v12 = (v12 >> 16) ^ (v12 << 48);
    v08 += v12;
    v04 ^= v08;
    v04 = (v04 >> 63) ^ (v04 <<  1);

    v01 += v05;
    v13 ^= v01;
    v13 = (v13 >> 16) ^ (v13 << 48);
    v09 += v13;
    v05 ^= v09;
    v05 = (v05 >> 63) ^ (v05 <<  1);

    v00 += m04;
    v00 += v05;
    v15 ^= v00;
    v15 = (v15 >> 32) ^ (v15 << 32);
    v10 += v15;
    v05 ^= v10;
    v05 = (v05 >> 24) ^ (v05 << 40);

    v01 += m07;
    v01 += v06;
    v12 ^= v01;
    v12 = (v12 >> 32) ^ (v12 << 32);
    v11 += v12;
    v06 ^= v11;
    v06 = (v06 >> 24) ^ (v06 << 40);

    v02 += v07;
    v13 ^= v02;
    v13 = (v13 >> 32) ^ (v13 << 32);
    v08 += v13;
    v07 ^= v08;
    v07 = (v07 >> 24) ^ (v07 << 40);

    v03 += m01;
    v03 += v04;
    v14 ^= v03;
    v14 = (v14 >> 32) ^ (v14 << 32);
    v09 += v14;
    v04 ^= v09;
    v04 = (v04 >> 24) ^ (v04 << 40);

    v02 += v07;
    v13 ^= v02;
    v13 = (v13 >> 16) ^ (v13 << 48);
    v08 += v13;
    v07 ^= v08;
    v07 = (v07 >> 63) ^ (v07 <<  1);

    v03 += v04;
    v14 ^= v03;
    v14 = (v14 >> 16) ^ (v14 << 48);
    v09 += v14;
    v04 ^= v09;
    v04 = (v04 >> 63) ^ (v04 <<  1);

    v00 += v05;
    v15 ^= v00;
    v15 = (v15 >> 16) ^ (v15 << 48);
    v10 += v15;
    v05 ^= v10;
    v05 = (v05 >> 63) ^ (v05 <<  1);

    v01 += m05;
    v01 += v06;
    v12 ^= v01;
    v12 = (v12 >> 16) ^ (v12 << 48);
    v11 += v12;
    v06 ^= v11;
    v06 = (v06 >> 63) ^ (v06 <<  1);

    //ROUND 7
    v00 += v04;
    v12 ^= v00;
    v12 = (v12 >> 32) ^ (v12 << 32);
    v08 += v12;
    v04 ^= v08;
    v04 = (v04 >> 24) ^ (v04 << 40);

    v01 += m01;
    v01 += v05;
    v13 ^= v01;
    v13 = (v13 >> 32) ^ (v13 << 32);
    v09 += v13;
    v05 ^= v09;
    v05 = (v05 >> 24) ^ (v05 << 40);

    v02 += v06;
    v14 ^= v02;
    v14 = (v14 >> 32) ^ (v14 << 32);
    v10 += v14;
    v06 ^= v10;
    v06 = (v06 >> 24) ^ (v06 << 40);

    v03 += m04;
    v03 += v07;
    v15 ^= v03;
    v15 = (v15 >> 32) ^ (v15 << 32);
    v11 += v15;
    v07 ^= v11;
    v07 = (v07 >> 24) ^ (v07 << 40);

    v02 += v06;
    v14 ^= v02;
    v14 = (v14 >> 16) ^ (v14 << 48);
    v10 += v14;
    v06 ^= v10;
    v06 = (v06 >> 63) ^ (v06 <<  1);

    v03 += v07;
    v15 ^= v03;
    v15 = (v15 >> 16) ^ (v15 << 48);
    v11 += v15;
    v07 ^= v11;
    v07 = (v07 >> 63) ^ (v07 <<  1);

    v00 += m05;
    v00 += v04;
    v12 ^= v00;
    v12 = (v12 >> 16) ^ (v12 << 48);
    v08 += v12;
    v04 ^= v08;
    v04 = (v04 >> 63) ^ (v04 <<  1);

    v01 += v05;
    v13 ^= v01;
    v13 = (v13 >> 16) ^ (v13 << 48);
    v09 += v13;
    v05 ^= v09;
    v05 = (v05 >> 63) ^ (v05 <<  1);

    v00 += m00;
    v00 += v05;
    v15 ^= v00;
    v15 = (v15 >> 32) ^ (v15 << 32);
    v10 += v15;
    v05 ^= v10;
    v05 = (v05 >> 24) ^ (v05 << 40);

    v01 += m06;
    v01 += v06;
    v12 ^= v01;
    v12 = (v12 >> 32) ^ (v12 << 32);
    v11 += v12;
    v06 ^= v11;
    v06 = (v06 >> 24) ^ (v06 << 40);

    v02 += v07;
    v13 ^= v02;
    v13 = (v13 >> 32) ^ (v13 << 32);
    v08 += v13;
    v07 ^= v08;
    v07 = (v07 >> 24) ^ (v07 << 40);

    v03 += v04;
    v14 ^= v03;
    v14 = (v14 >> 32) ^ (v14 << 32);
    v09 += v14;
    v04 ^= v09;
    v04 = (v04 >> 24) ^ (v04 << 40);

    v02 += m02;
    v02 += v07;
    v13 ^= v02;
    v13 = (v13 >> 16) ^ (v13 << 48);
    v08 += v13;
    v07 ^= v08;
    v07 = (v07 >> 63) ^ (v07 <<  1);

    v03 += v04;
    v14 ^= v03;
    v14 = (v14 >> 16) ^ (v14 << 48);
    v09 += v14;
    v04 ^= v09;
    v04 = (v04 >> 63) ^ (v04 <<  1);

    v00 += m07;
    v00 += v05;
    v15 ^= v00;
    v15 = (v15 >> 16) ^ (v15 << 48);
    v10 += v15;
    v05 ^= v10;
    v05 = (v05 >> 63) ^ (v05 <<  1);

    v01 += m03;
    v01 += v06;
    v12 ^= v01;
    v12 = (v12 >> 16) ^ (v12 << 48);
    v11 += v12;
    v06 ^= v11;
    v06 = (v06 >> 63) ^ (v06 <<  1);

    //ROUND 8
    v00 += v04;
    v12 ^= v00;
    v12 = (v12 >> 32) ^ (v12 << 32);
    v08 += v12;
    v04 ^= v08;
    v04 = (v04 >> 24) ^ (v04 << 40);

    v01 += m07;
    v01 += v05;
    v13 ^= v01;
    v13 = (v13 >> 32) ^ (v13 << 32);
    v09 += v13;
    v05 ^= v09;
    v05 = (v05 >> 24) ^ (v05 << 40);

    v02 += v06;
    v14 ^= v02;
    v14 = (v14 >> 32) ^ (v14 << 32);
    v10 += v14;
    v06 ^= v10;
    v06 = (v06 >> 24) ^ (v06 << 40);

    v03 += m03;
    v03 += v07;
    v15 ^= v03;
    v15 = (v15 >> 32) ^ (v15 << 32);
    v11 += v15;
    v07 ^= v11;
    v07 = (v07 >> 24) ^ (v07 << 40);

    v02 += m01;
    v02 += v06;
    v14 ^= v02;
    v14 = (v14 >> 16) ^ (v14 << 48);
    v10 += v14;
    v06 ^= v10;
    v06 = (v06 >> 63) ^ (v06 <<  1);

    v03 += v07;
    v15 ^= v03;
    v15 = (v15 >> 16) ^ (v15 << 48);
    v11 += v15;
    v07 ^= v11;
    v07 = (v07 >> 63) ^ (v07 <<  1);

    v00 += v04;
    v12 ^= v00;
    v12 = (v12 >> 16) ^ (v12 << 48);
    v08 += v12;
    v04 ^= v08;
    v04 = (v04 >> 63) ^ (v04 <<  1);

    v01 += v05;
    v13 ^= v01;
    v13 = (v13 >> 16) ^ (v13 << 48);
    v09 += v13;
    v05 ^= v09;
    v05 = (v05 >> 63) ^ (v05 <<  1);

    v00 += m05;
    v00 += v05;
    v15 ^= v00;
    v15 = (v15 >> 32) ^ (v15 << 32);
    v10 += v15;
    v05 ^= v10;
    v05 = (v05 >> 24) ^ (v05 << 40);

    v01 += v06;
    v12 ^= v01;
    v12 = (v12 >> 32) ^ (v12 << 32);
    v11 += v12;
    v06 ^= v11;
    v06 = (v06 >> 24) ^ (v06 << 40);

    v02 += v07;
    v13 ^= v02;
    v13 = (v13 >> 32) ^ (v13 << 32);
    v08 += v13;
    v07 ^= v08;
    v07 = (v07 >> 24) ^ (v07 << 40);

    v03 += m02;
    v03 += v04;
    v14 ^= v03;
    v14 = (v14 >> 32) ^ (v14 << 32);
    v09 += v14;
    v04 ^= v09;
    v04 = (v04 >> 24) ^ (v04 << 40);

    v02 += m06;
    v02 += v07;
    v13 ^= v02;
    v13 = (v13 >> 16) ^ (v13 << 48);
    v08 += v13;
    v07 ^= v08;
    v07 = (v07 >> 63) ^ (v07 <<  1);

    v03 += v04;
    v14 ^= v03;
    v14 = (v14 >> 16) ^ (v14 << 48);
    v09 += v14;
    v04 ^= v09;
    v04 = (v04 >> 63) ^ (v04 <<  1);

    v00 += m00;
    v00 += v05;
    v15 ^= v00;
    v15 = (v15 >> 16) ^ (v15 << 48);
    v10 += v15;
    v05 ^= v10;
    v05 = (v05 >> 63) ^ (v05 <<  1);

    v01 += m04;
    v01 += v06;
    v12 ^= v01;
    v12 = (v12 >> 16) ^ (v12 << 48);
    v11 += v12;
    v06 ^= v11;
    v06 = (v06 >> 63) ^ (v06 <<  1);

    //ROUND 9
    v00 += m06;
    v00 += v04;
    v12 ^= v00;
    v12 = (v12 >> 32) ^ (v12 << 32);
    v08 += v12;
    v04 ^= v08;
    v04 = (v04 >> 24) ^ (v04 << 40);

    v01 += v05;
    v13 ^= v01;
    v13 = (v13 >> 32) ^ (v13 << 32);
    v09 += v13;
    v05 ^= v09;
    v05 = (v05 >> 24) ^ (v05 << 40);

    v02 += v06;
    v14 ^= v02;
    v14 = (v14 >> 32) ^ (v14 << 32);
    v10 += v14;
    v06 ^= v10;
    v06 = (v06 >> 24) ^ (v06 << 40);

    v03 += m00;
    v03 += v07;
    v15 ^= v03;
    v15 = (v15 >> 32) ^ (v15 << 32);
    v11 += v15;
    v07 ^= v11;
    v07 = (v07 >> 24) ^ (v07 << 40);

    v02 += m03;
    v02 += v06;
    v14 ^= v02;
    v14 = (v14 >> 16) ^ (v14 << 48);
    v10 += v14;
    v06 ^= v10;
    v06 = (v06 >> 63) ^ (v06 <<  1);

    v03 += v07;
    v15 ^= v03;
    v15 = (v15 >> 16) ^ (v15 << 48);
    v11 += v15;
    v07 ^= v11;
    v07 = (v07 >> 63) ^ (v07 <<  1);

    v00 += v04;
    v12 ^= v00;
    v12 = (v12 >> 16) ^ (v12 << 48);
    v08 += v12;
    v04 ^= v08;
    v04 = (v04 >> 63) ^ (v04 <<  1);

    v01 += v05;
    v13 ^= v01;
    v13 = (v13 >> 16) ^ (v13 << 48);
    v09 += v13;
    v05 ^= v09;
    v05 = (v05 >> 63) ^ (v05 <<  1);

    v00 += v05;
    v15 ^= v00;
    v15 = (v15 >> 32) ^ (v15 << 32);
    v10 += v15;
    v05 ^= v10;
    v05 = (v05 >> 24) ^ (v05 << 40);

    v01 += v06;
    v12 ^= v01;
    v12 = (v12 >> 32) ^ (v12 << 32);
    v11 += v12;
    v06 ^= v11;
    v06 = (v06 >> 24) ^ (v06 << 40);

    v02 += m01;
    v02 += v07;
    v13 ^= v02;
    v13 = (v13 >> 32) ^ (v13 << 32);
    v08 += v13;
    v07 ^= v08;
    v07 = (v07 >> 24) ^ (v07 << 40);

    v03 += v04;
    v14 ^= v03;
    v14 = (v14 >> 32) ^ (v14 << 32);
    v09 += v14;
    v04 ^= v09;
    v04 = (v04 >> 24) ^ (v04 << 40);

    v02 += m04;
    v02 += v07;
    v13 ^= v02;
    v13 = (v13 >> 16) ^ (v13 << 48);
    v08 += v13;
    v07 ^= v08;
    v07 = (v07 >> 63) ^ (v07 <<  1);

    v03 += m05;
    v03 += v04;
    v14 ^= v03;
    v14 = (v14 >> 16) ^ (v14 << 48);
    v09 += v14;
    v04 ^= v09;
    v04 = (v04 >> 63) ^ (v04 <<  1);

    v00 += m02;
    v00 += v05;
    v15 ^= v00;
    v15 = (v15 >> 16) ^ (v15 << 48);
    v10 += v15;
    v05 ^= v10;
    v05 = (v05 >> 63) ^ (v05 <<  1);

    v01 += m07;
    v01 += v06;
    v12 ^= v01;
    v12 = (v12 >> 16) ^ (v12 << 48);
    v11 += v12;
    v06 ^= v11;
    v06 = (v06 >> 63) ^ (v06 <<  1);

    //ROUND 10
    v00 += v04;
    v12 ^= v00;
    v12 = (v12 >> 32) ^ (v12 << 32);
    v08 += v12;
    v04 ^= v08;
    v04 = (v04 >> 24) ^ (v04 << 40);

    v01 += v05;
    v13 ^= v01;
    v13 = (v13 >> 32) ^ (v13 << 32);
    v09 += v13;
    v05 ^= v09;
    v05 = (v05 >> 24) ^ (v05 << 40);

    v02 += m07;
    v02 += v06;
    v14 ^= v02;
    v14 = (v14 >> 32) ^ (v14 << 32);
    v10 += v14;
    v06 ^= v10;
    v06 = (v06 >> 24) ^ (v06 << 40);

    v03 += m01;
    v03 += v07;
    v15 ^= v03;
    v15 = (v15 >> 32) ^ (v15 << 32);
    v11 += v15;
    v07 ^= v11;
    v07 = (v07 >> 24) ^ (v07 << 40);

    v02 += m06;
    v02 += v06;
    v14 ^= v02;
    v14 = (v14 >> 16) ^ (v14 << 48);
    v10 += v14;
    v06 ^= v10;
    v06 = (v06 >> 63) ^ (v06 <<  1);

    v03 += m05;
    v03 += v07;
    v15 ^= v03;
    v15 = (v15 >> 16) ^ (v15 << 48);
    v11 += v15;
    v07 ^= v11;
    v07 = (v07 >> 63) ^ (v07 <<  1);

    v00 += m02;
    v00 += v04;
    v12 ^= v00;
    v12 = (v12 >> 16) ^ (v12 << 48);
    v08 += v12;
    v04 ^= v08;
    v04 = (v04 >> 63) ^ (v04 <<  1);

    v01 += m04;
    v01 += v05;
    v13 ^= v01;
    v13 = (v13 >> 16) ^ (v13 << 48);
    v09 += v13;
    v05 ^= v09;
    v05 = (v05 >> 63) ^ (v05 <<  1);

    v00 += v05;
    v15 ^= v00;
    v15 = (v15 >> 32) ^ (v15 << 32);
    v10 += v15;
    v05 ^= v10;
    v05 = (v05 >> 24) ^ (v05 << 40);

    v01 += v06;
    v12 ^= v01;
    v12 = (v12 >> 32) ^ (v12 << 32);
    v11 += v12;
    v06 ^= v11;
    v06 = (v06 >> 24) ^ (v06 << 40);

    v02 += m03;
    v02 += v07;
    v13 ^= v02;
    v13 = (v13 >> 32) ^ (v13 << 32);
    v08 += v13;
    v07 ^= v08;
    v07 = (v07 >> 24) ^ (v07 << 40);

    v03 += v04;
    v14 ^= v03;
    v14 = (v14 >> 32) ^ (v14 << 32);
    v09 += v14;
    v04 ^= v09;
    v04 = (v04 >> 24) ^ (v04 << 40);

    v02 += v07;
    v13 ^= v02;
    v13 = (v13 >> 16) ^ (v13 << 48);
    v08 += v13;
    v07 ^= v08;
    v07 = (v07 >> 63) ^ (v07 <<  1);

    v03 += m00;
    v03 += v04;
    v14 ^= v03;
    v14 = (v14 >> 16) ^ (v14 << 48);
    v09 += v14;
    v04 ^= v09;
    v04 = (v04 >> 63) ^ (v04 <<  1);

    v00 += v05;
    v15 ^= v00;
    v15 = (v15 >> 16) ^ (v15 << 48);
    v10 += v15;
    v05 ^= v10;
    v05 = (v05 >> 63) ^ (v05 <<  1);

    v01 += v06;
    v12 ^= v01;
    v12 = (v12 >> 16) ^ (v12 << 48);
    v11 += v12;
    v06 ^= v11;
    v06 = (v06 >> 63) ^ (v06 <<  1);

    //ROUND 11
    v00 += m00;
    v00 += v04;
    v12 ^= v00;
    v12 = (v12 >> 32) ^ (v12 << 32);
    v08 += v12;
    v04 ^= v08;
    v04 = (v04 >> 24) ^ (v04 << 40);

    v01 += m02;
    v01 += v05;
    v13 ^= v01;
    v13 = (v13 >> 32) ^ (v13 << 32);
    v09 += v13;
    v05 ^= v09;
    v05 = (v05 >> 24) ^ (v05 << 40);

    v02 += m04;
    v02 += v06;
    v14 ^= v02;
    v14 = (v14 >> 32) ^ (v14 << 32);
    v10 += v14;
    v06 ^= v10;
    v06 = (v06 >> 24) ^ (v06 << 40);

    v03 += m06;
    v03 += v07;
    v15 ^= v03;
    v15 = (v15 >> 32) ^ (v15 << 32);
    v11 += v15;
    v07 ^= v11;
    v07 = (v07 >> 24) ^ (v07 << 40);

    v02 += m05;
    v02 += v06;
    v14 ^= v02;
    v14 = (v14 >> 16) ^ (v14 << 48);
    v10 += v14;
    v06 ^= v10;
    v06 = (v06 >> 63) ^ (v06 <<  1);

    v03 += m07;
    v03 += v07;
    v15 ^= v03;
    v15 = (v15 >> 16) ^ (v15 << 48);
    v11 += v15;
    v07 ^= v11;
    v07 = (v07 >> 63) ^ (v07 <<  1);

    v00 += m01;
    v00 += v04;
    v12 ^= v00;
    v12 = (v12 >> 16) ^ (v12 << 48);
    v08 += v12;
    v04 ^= v08;
    v04 = (v04 >> 63) ^ (v04 <<  1);

    v01 += m03;
    v01 += v05;
    v13 ^= v01;
    v13 = (v13 >> 16) ^ (v13 << 48);
    v09 += v13;
    v05 ^= v09;
    v05 = (v05 >> 63) ^ (v05 <<  1);

    v00 += v05;
    v15 ^= v00;
    v15 = (v15 >> 32) ^ (v15 << 32);
    v10 += v15;
    v05 ^= v10;
    v05 = (v05 >> 24) ^ (v05 << 40);

    v01 += v06;
    v12 ^= v01;
    v12 = (v12 >> 32) ^ (v12 << 32);
    v11 += v12;
    v06 ^= v11;
    v06 = (v06 >> 24) ^ (v06 << 40);

    v02 += v07;
    v13 ^= v02;
    v13 = (v13 >> 32) ^ (v13 << 32);
    v08 += v13;
    v07 ^= v08;
    v07 = (v07 >> 24) ^ (v07 << 40);

    v03 += v04;
    v14 ^= v03;
    v14 = (v14 >> 32) ^ (v14 << 32);
    v09 += v14;
    v04 ^= v09;
    v04 = (v04 >> 24) ^ (v04 << 40);

    v02 += v07;
    v13 ^= v02;
    v13 = (v13 >> 16) ^ (v13 << 48);
    v08 += v13;
    v07 ^= v08;
    v07 = (v07 >> 63) ^ (v07 <<  1);

    v03 += v04;
    v14 ^= v03;
    v14 = (v14 >> 16) ^ (v14 << 48);
    v09 += v14;
    v04 ^= v09;
    v04 = (v04 >> 63) ^ (v04 <<  1);

    v00 += v05;
    v15 ^= v00;
    v15 = (v15 >> 16) ^ (v15 << 48);
    v10 += v15;
    v05 ^= v10;
    v05 = (v05 >> 63) ^ (v05 <<  1);

    v01 += v06;
    v12 ^= v01;
    v12 = (v12 >> 16) ^ (v12 << 48);
    v11 += v12;
    v06 ^= v11;
    v06 = (v06 >> 63) ^ (v06 <<  1);

    //ROUND 12
    v00 += v04;
    v12 ^= v00;
    v12 = (v12 >> 32) ^ (v12 << 32);
    v08 += v12;
    v04 ^= v08;
    v04 = (v04 >> 24) ^ (v04 << 40);

    v01 += m04;
    v01 += v05;
    v13 ^= v01;
    v13 = (v13 >> 32) ^ (v13 << 32);
    v09 += v13;
    v05 ^= v09;
    v05 = (v05 >> 24) ^ (v05 << 40);

    v02 += v06;
    v14 ^= v02;
    v14 = (v14 >> 32) ^ (v14 << 32);
    v10 += v14;
    v06 ^= v10;
    v06 = (v06 >> 24) ^ (v06 << 40);

    v03 += v07;
    v15 ^= v03;
    v15 = (v15 >> 32) ^ (v15 << 32);
    v11 += v15;
    v07 ^= v11;
    v07 = (v07 >> 24) ^ (v07 << 40);

    v02 += v06;
    v14 ^= v02;
    v14 = (v14 >> 16) ^ (v14 << 48);
    v10 += v14;
    v06 ^= v10;
    v06 = (v06 >> 63) ^ (v06 <<  1);

    v03 += m06;
    v03 += v07;
    v15 ^= v03;
    v15 = (v15 >> 16) ^ (v15 << 48);
    v11 += v15;
    v07 ^= v11;
    v07 = (v07 >> 63) ^ (v07 <<  1);

    v00 += v04;
    v12 ^= v00;
    v12 = (v12 >> 16) ^ (v12 << 48);
    v08 += v12;
    v04 ^= v08;
    v04 = (v04 >> 63) ^ (v04 <<  1);

    v01 += v05;
    v13 ^= v01;
    v13 = (v13 >> 16) ^ (v13 << 48);
    v09 += v13;
    v05 ^= v09;
    v05 = (v05 >> 63) ^ (v05 <<  1);

    v00 += m01;
    v00 += v05;
    v15 ^= v00;
    v15 = (v15 >> 32) ^ (v15 << 32);
    v10 += v15;
    v05 ^= v10;
    v05 = (v05 >> 24) ^ (v05 << 40);

    v01 += m00;
    v01 += v06;
    v12 ^= v01;
    v12 = (v12 >> 32) ^ (v12 << 32);
    v11 += v12;
    v06 ^= v11;
    v06 = (v06 >> 24) ^ (v06 << 40);

    v02 += v07;
    v13 ^= v02;
    v13 = (v13 >> 32) ^ (v13 << 32);
    v08 += v13;
    v07 ^= v08;
    v07 = (v07 >> 24) ^ (v07 << 40);

    v03 += m05;
    v03 += v04;
    v14 ^= v03;
    v14 = (v14 >> 32) ^ (v14 << 32);
    v09 += v14;
    v04 ^= v09;
    v04 = (v04 >> 24) ^ (v04 << 40);

    v02 += m07;
    v02 += v07;
    v13 ^= v02;
    v13 = (v13 >> 16) ^ (v13 << 48);
    v08 += v13;
    v07 ^= v08;
    v07 = (v07 >> 63) ^ (v07 <<  1);

    v03 += m03;
    v03 += v04;
    v14 ^= v03;
    v14 = (v14 >> 16) ^ (v14 << 48);
    v09 += v14;
    v04 ^= v09;
    v04 = (v04 >> 63) ^ (v04 <<  1);

    v00 += v05;
    v15 ^= v00;
    v15 = (v15 >> 16) ^ (v15 << 48);
    v10 += v15;
    v05 ^= v10;
    v05 = (v05 >> 63) ^ (v05 <<  1);

    v01 += m02;
    v01 += v06;
    v12 ^= v01;
    v12 = (v12 >> 16) ^ (v12 << 48);
    v11 += v12;
    v06 ^= v11;
    v06 = (v06 >> 63) ^ (v06 <<  1);

    sh[0] = v00 ^ v08 ^ iv0 ^ 0x01010020;
    sh[1] = v01 ^ v09 ^ iv1;
    sh[2] = v02 ^ v10 ^ iv2;
    sh[3] = v03 ^ v11 ^ iv3;
}

kernel void merkle(device uchar* data, constant uint& round, constant uint& leaves, uint id [[thread_position_in_grid]])
{
    device  ulong* lhs = (device  ulong*)&data[DIGEST_SIZE * (id << (round + 1))];
    device  ulong* rhs = (device  ulong*)&data[DIGEST_SIZE * ((id << (round + 1)) + (1 << round))];
    
    ulong m[8] = { lhs[0], lhs[1], lhs[2], lhs[3], rhs[0], rhs[1], rhs[2], rhs[3] };
    
    ::mix_scalar(lhs, m);
}
