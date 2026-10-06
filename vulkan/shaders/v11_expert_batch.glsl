layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer RamWeights {
    uint ram_words[];
};
layout(set = 0, binding = 1, std430) readonly buffer VramWeights {
    uint vram_words[];
};
layout(set = 0, binding = 2, std430) readonly buffer Q8Input {
    uint q8[];
};
layout(set = 0, binding = 3, std430) writeonly buffer Output {
    float y[];
};
layout(set = 0, binding = 4, std430) readonly buffer ExpertMeta {
    uint meta[];
};

layout(set = 0, binding = 5, std430) readonly buffer IQ2XXSGrid {
    uvec2 iq2xxs_grid[];
};
layout(set = 0, binding = 6, std430) readonly buffer IQ2XSGrid {
    uvec2 iq2xs_grid[];
};
layout(set = 0, binding = 7, std430) readonly buffer IQ2SGrid {
    uvec2 iq2s_grid[];
};
layout(set = 0, binding = 8, std430) readonly buffer IQ3XXSGrid {
    uint iq3xxs_grid[];
};
layout(set = 0, binding = 9, std430) readonly buffer IQ3SGrid {
    uint iq3s_grid[];
};
layout(set = 0, binding = 10, std430) readonly buffer KSigns {
    uint ksigns[];
};
layout(set = 0, binding = 11, std430) readonly buffer IQ4Values {
    int iq4_values[];
};

layout(push_constant) uniform Push {
    uint k;
    uint n_embd;
    uint n_ff;
    uint role;
    uint rows;
    uint input_stride;
    uint output_stride;
    uint reserved;
} pc;

layout(set=0,binding=12,std430) readonly buffer GroupSlots {uint slots[];};
layout(constant_id=0) const uint COLUMNS=5;
shared float reduce_buf[32*COLUMNS];

uint m(uint rank, uint field) {
    return meta[rank * 8u + field];
}

uint tier_word(uint tier, uint word_index) {
    return tier == 0u ? ram_words[word_index] : vram_words[word_index];
}

uint load_u8(uint tier, uint off) {
    uint w = tier_word(tier, off >> 2u);
    return (w >> ((off & 3u) * 8u)) & 0xffu;
}

// Explicit unaligned word fetch.
// Compared with four load_u8() calls this guarantees at most two uint loads.
uint load_u32_fast(uint tier, uint off) {
    uint wi = off >> 2u;
    uint sh = (off & 3u) * 8u;
    uint lo = tier_word(tier, wi);
    if (sh == 0u) return lo;
    uint hi = tier_word(tier, wi + 1u);
    return (lo >> sh) | (hi << (32u - sh));
}

uint load_u16_fast(uint tier, uint off) {
    return load_u32_fast(tier, off) & 0xffffu;
}

float load_f16(uint tier, uint off) {
    return unpackHalf2x16(load_u16_fast(tier, off)).x;
}

uint byte32(uint v, uint j) {
    return (v >> (8u * j)) & 0xffu;
}

uint pack_s8_4(int a, int b, int c, int d) {
    return (uint(a) & 0xffu) |
           ((uint(b) & 0xffu) << 8u) |
           ((uint(c) & 0xffu) << 16u) |
           ((uint(d) & 0xffu) << 24u);
}

int dp4a_hw(uint a, uint b) {
    return dotPacked4x8EXT(int(a), int(b));
}

uint q8_block_base(uint q8_base_words, uint value_index) {
    return q8_base_words + (value_index >> 5u) * 9u;
}

float q8_scale(uint q8_base_words, uint value_index) {
    return unpackHalf2x16(q8[q8_block_base(q8_base_words, value_index)]).x;
}

uint q8_word4(uint q8_base_words, uint value_index) {
    uint block_base = q8_block_base(q8_base_words, value_index);
    uint lane4 = value_index & 31u;
    return q8[block_base + 1u + (lane4 >> 2u)];
}

uint signed_grid4(uint raw, uint sign_bits, uint sign_shift) {
    int a = int(byte32(raw, 0u));
    int b = int(byte32(raw, 1u));
    int c = int(byte32(raw, 2u));
    int d = int(byte32(raw, 3u));
    if ((sign_bits & (1u << (sign_shift + 0u))) != 0u) a = -a;
    if ((sign_bits & (1u << (sign_shift + 1u))) != 0u) b = -b;
    if ((sign_bits & (1u << (sign_shift + 2u))) != 0u) c = -c;
    if ((sign_bits & (1u << (sign_shift + 3u))) != 0u) d = -d;
    return pack_s8_4(a, b, c, d);
}

int dot8_grid(uvec2 g, uint signs, uint q8_base_words, uint value_index) {
    uint w0 = signed_grid4(g.x, signs, 0u);
    uint w1 = signed_grid4(g.y, signs, 4u);
    return dp4a_hw(w0, q8_word4(q8_base_words, value_index)) +
           dp4a_hw(w1, q8_word4(q8_base_words, value_index + 4u));
}

int dot8_iq3(uint g0, uint g1, uint signs,
             uint q8_base_words, uint value_index) {
    uint w0 = signed_grid4(g0, signs, 0u);
    uint w1 = signed_grid4(g1, signs, 4u);
    return dp4a_hw(w0, q8_word4(q8_base_words, value_index)) +
           dp4a_hw(w1, q8_word4(q8_base_words, value_index + 4u));
}

uint iq4_word_raw(uint raw, bool high) {
    int a = iq4_values[high ? (byte32(raw, 0u) >> 4u) : (byte32(raw, 0u) & 0xfu)];
    int b = iq4_values[high ? (byte32(raw, 1u) >> 4u) : (byte32(raw, 1u) & 0xfu)];
    int c = iq4_values[high ? (byte32(raw, 2u) >> 4u) : (byte32(raw, 2u) & 0xfu)];
    int d = iq4_values[high ? (byte32(raw, 3u) >> 4u) : (byte32(raw, 3u) & 0xfu)];
    return pack_s8_4(a, b, c, d);
}

uint q2_codes4(uint packed) {
    int a = int((packed >> 0u) & 3u) - 1;
    int b = int((packed >> 2u) & 3u) - 1;
    int c = int((packed >> 4u) & 3u) - 1;
    int d = int((packed >> 6u) & 3u) - 1;
    return pack_s8_4(a, b, c, d);
}

uint bases[COLUMNS];uint destinations[COLUMNS];float accum[COLUMNS], input_scales[COLUMNS];
void input_scale(uint vi){[[unroll]] for(uint t=0;t<COLUMNS;t++)if(t<pc.rows&&destinations[t]!=0xffffffffu)input_scales[t]=q8_scale(bases[t],vi);}
void add_part(float dw,uvec2 codes,uint vi){
 [[unroll]] for(uint t=0;t<COLUMNS;t++)if(t<pc.rows&&destinations[t]!=0xffffffffu){int dotv=dp4a_hw(codes.x,q8_word4(bases[t],vi))+dp4a_hw(codes.y,q8_word4(bases[t],vi+4));accum[t]+=(dw*input_scales[t])*float(dotv);}
}

void main(){uint lane=gl_LocalInvocationID.x,sub=lane&7,row_lane=lane/8,rank=gl_WorkGroupID.y,row_global=gl_WorkGroupID.x*4+row_lane;
 bool valid=row_global<(pc.role==0?2*pc.n_ff:pc.n_embd);uint tier=m(rank,0),row=row_global;bool up=pc.role==0&&row>=pc.n_ff;if(up)row-=pc.n_ff;
 uint base=pc.role==0?(up?m(rank,4):m(rank,3)):m(rank,5),rowbytes=pc.role==0?m(rank,6):m(rank,7),cols=pc.role==0?pc.n_embd:pc.n_ff,row0=base+row*rowbytes;
 [[unroll]] for(uint t=0;t<COLUMNS;t++){uint slot=slots[rank*5+t];destinations[t]=slot;bases[t]=pc.role==0?t*pc.input_stride:t*pc.input_stride+(slot%10)*(pc.n_ff/32)*9;accum[t]=0;}
 if(valid){
#if JR_TYPE==16 || JR_TYPE==17 || JR_TYPE==18 || JR_TYPE==21 || JR_TYPE==22
 for(uint b=0;b<cols/256;b++){
  uint sb=row0+b*(JR_TYPE==16?66:JR_TYPE==17?74:JR_TYPE==18?98:JR_TYPE==21?110:82),ib=sub;float d0=load_f16(tier,sb);
  uint qs=sb+2;
#if JR_TYPE==16
  uint qb=qs+8*ib,aux=load_u32_fast(tier,qb+4),packed_grids=load_u32_fast(tier,qb);float block_dw=d0*(0.5+float(aux>>28))*0.25;
#elif JR_TYPE==18
  uint aux=load_u32_fast(tier,qs+64+4*ib);
  uvec2 packed_pairs=uvec2(load_u32_fast(tier,qs+8*ib),load_u32_fast(tier,qs+8*ib+4));float block_dw=d0*(0.5+float(aux>>28))*0.5;
#elif JR_TYPE==21
  uint hi=load_u8(tier,qs+64+ib),sc=load_u8(tier,qs+104+(ib>>1)),packed_signs=load_u32_fast(tier,qs+72+4*ib);
  uvec2 packed_pairs=uvec2(load_u32_fast(tier,qs+8*ib),load_u32_fast(tier,qs+8*ib+4));float block_dw=d0*float(1+2*((sc>>(4*(ib&1)))&15));
#elif JR_TYPE==17
  uint sc=load_u8(tier,qs+64+ib);
  uvec2 packed_pairs=uvec2(load_u32_fast(tier,qs+8*ib),load_u32_fast(tier,qs+8*ib+4));
#elif JR_TYPE==22
  uint hi=load_u8(tier,qs+64+ib),sc=load_u8(tier,qs+72+ib),packed_grids=load_u32_fast(tier,qs+4*ib),packed_signs=load_u32_fast(tier,qs+32+4*ib);
#endif
  input_scale(b*256+32*ib);
  for(uint il=0;il<4;il++){uint vi=b*256+32*ib+8*il;float dw;uvec2 codes;
#if JR_TYPE==16
   uint gi=(packed_grids>>(8*il))&255,signs=ksigns[(aux>>(7*il))&127]&255;uvec2 grid=iq2xxs_grid[gi];codes=uvec2(signed_grid4(grid.x,signs,0),signed_grid4(grid.y,signs,4));dw=block_dw;
#elif JR_TYPE==17
   uint q=(packed_pairs[il>>1]>>((il&1)*16))&65535,signs=ksigns[q>>9]&255;uvec2 grid=iq2xs_grid[q&511];codes=uvec2(signed_grid4(grid.x,signs,0),signed_grid4(grid.y,signs,4));dw=d0*(0.5+float((sc>>(4*(il>>1)))&15))*0.25;
#elif JR_TYPE==22
   uint qi=(packed_grids>>(8*il))&255,gi=qi|((hi<<(8-2*il))&0x300),signs=(packed_signs>>(8*il))&255;uvec2 grid=iq2s_grid[gi];codes=uvec2(signed_grid4(grid.x,signs,0),signed_grid4(grid.y,signs,4));dw=d0*(0.5+float((sc>>(4*(il>>1)))&15))*0.25;
#elif JR_TYPE==18
   uint pair=(packed_pairs[il>>1]>>((il&1)*16))&65535,signs=ksigns[(aux>>(7*il))&127]&255;codes=uvec2(signed_grid4(iq3xxs_grid[pair&255],signs,0),signed_grid4(iq3xxs_grid[pair>>8],signs,4));dw=block_dw;
#elif JR_TYPE==21
   uint pair=(packed_pairs[il>>1]>>((il&1)*16))&65535,signs=(packed_signs>>(8*il))&255;uint idx0=(pair&255)|((hi<<(8-2*il))&0x100),idx1=(pair>>8)|((hi<<(7-2*il))&0x100);codes=uvec2(signed_grid4(iq3s_grid[idx0],signs,0),signed_grid4(iq3s_grid[idx1],signs,4));dw=block_dw;
#endif
   add_part(dw,codes,vi);
  }
 }
#elif JR_TYPE==20 || JR_TYPE==42
 uint blocksize=JR_TYPE==20?32:64,nb=cols/blocksize;
 for(uint c=0;c<(nb+7)/8;c++){uint bi=c*8+sub;if(bi<nb){uint wb=row0+bi*18;float dw=load_f16(tier,wb);uint qs=wb+2;uvec4 raw=uvec4(load_u32_fast(tier,qs),load_u32_fast(tier,qs+4),load_u32_fast(tier,qs+8),load_u32_fast(tier,qs+12));
#if JR_TYPE==20
 input_scale(bi*32);
 for(uint il=0;il<4;il++){bool high=il>=2,upper=(il&1)!=0;uint a=upper?raw.z:raw.x,b=upper?raw.w:raw.y;add_part(dw,uvec2(iq4_word_raw(a,high),iq4_word_raw(b,high)),bi*32+il*8);}
#else
 for(uint part=0;part<8;part++){if((part&3)==0)input_scale(bi*64+part*8);uint pair=(raw[part>>1]>>((part&1)*16))&65535;add_part(dw,uvec2(q2_codes4(pair&255),q2_codes4(pair>>8)),bi*64+part*8);}
#endif
 }}
#endif
 }
 [[unroll]] for(uint t=0;t<COLUMNS;t++)reduce_buf[t*32+lane]=accum[t];barrier();
 for(uint step=4;step>0;step>>=1){if(sub<step)[[unroll]] for(uint t=0;t<COLUMNS;t++)reduce_buf[t*32+row_lane*8+sub]+=reduce_buf[t*32+row_lane*8+sub+step];barrier();}
 if(valid&&sub==0)[[unroll]] for(uint t=0;t<COLUMNS;t++)if(destinations[t]!=0xffffffffu){uint slot=destinations[t]%10,out_index=t*pc.output_stride+(pc.role==0?slot*2*pc.n_ff+row_global:slot*pc.n_embd+row);y[out_index]=reduce_buf[t*32+row_lane*8];}
}
