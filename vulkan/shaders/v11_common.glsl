// Swift native dense formats. Byte layouts follow strata/artifact/dequant.hpp
// and the validated V8 IQ3_S shader. All offsets below are bytes.
layout(set=0,binding=0,std430) readonly buffer Weights { uint w[]; };
layout(set=0,binding=1,std430) buffer Scratch { float s[]; };
layout(set=0,binding=2,std430) readonly buffer Control { uint ctl[]; };
layout(set=0,binding=3,std430) buffer State { float st[]; };
layout(set=0,binding=4,std430) buffer KV { uint kv[]; };
layout(set=0,binding=5,std430) buffer Diagnostics { float diag[]; };
layout(set=0,binding=5,std430) buffer DiagnosticBits { uint dbits[]; };
layout(set=0,binding=6,std430) readonly buffer IQ3Grid { uint grid[]; };
layout(set=0,binding=7,std430) readonly buffer Plan { uint plan[]; };
layout(set=0,binding=8,std430) buffer RouteMeta { uint meta[]; };
layout(set=0,binding=9,std430) buffer RouteWeights { float rw[]; };
layout(push_constant) uniform Push { uint op,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o; } p;
layout(set=0,binding=10,std430) buffer Snapshots { float snaps[]; };
layout(set=0,binding=11,std430) readonly buffer DraftVocab { uint dvocab[]; };
uint u8(uint off) {return (w[off/4]>>((off%4)*8))&255;}
uint u16(uint off) {return (w[off/4]>>((off%4)*8))&65535;}
float f16(uint off){return unpackHalf2x16(u16(off)).x;}
int i8(uint off){return int(u8(off)<<24)>>24;}
const int cb[16]=int[16](-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113);
float val(uint base,uint type,uint i) {
 if(type==0)return uintBitsToFloat(w[base/4+i]);
 if(type==30)return uintBitsToFloat(u16(base+2*i)<<16);
 if(type==1)return f16(base+2*i);
 if(type==8){uint off=base+(i/32)*34;return f16(off)*float(i8(off+2+i%32));}
 if(type==42){uint off=base+(i/64)*18;return f16(off)*float(int((u8(off+2+(i%64)/4)>>((i%4)*2))&3)-1);}
 if(type==20){uint off=base+(i/32)*18,j=i%32;return f16(off)*float(cb[(u8(off+2+j%16)>>((j/16)*4))&15]);}
 if(type==23){uint off=base+(i/256)*136,j=i%256,ib=j/32;
  int scale=int(((u8(off+4+ib/2)>>((ib%2)*4))&15)|(((u16(off+2)>>(2*ib))&3)<<4))-32;
  return f16(off)*float(scale)*float(cb[(u8(off+8+ib*16+j%16)>>(((j%32)/16)*4))&15]);}
 if(type==21){uint off=base+(i/256)*110,j=i%256,ib=j/32,il=(j%32)/8,q=(j%8)/4;
  uint gi=u8(off+2+ib*8+il*2+q)|(((u8(off+66+ib)>>(il*2+q))&1)<<8);
  int sym=int((grid[gi]>>((j%4)*8))&255);if(((u8(off+74+ib*4+il)>>(j%8))&1)!=0)sym=-sym;
  uint scale=(u8(off+106+ib/2)>>((ib%2)*4))&15;return f16(off)*float(1+2*scale)*float(sym);}
 if(type==14){uint off=base+(i/256)*210,j=i%256,halfb=j/128,z=j%128,l=z%32,grp=z/32;
  uint lo=u8(off+halfb*64+(grp%2)*32+l);uint hi=u8(off+128+halfb*32+l);
  int q=int(((lo>>((grp/2)*4))&15)|(((hi>>(2*grp))&3)<<4))-32;
  return f16(off+208)*float(i8(off+192+halfb*8+grp*2+l/16))*float(q);}
 if(type==12||type==13){uint bytes=type==12?144:176,off=base+(i/256)*bytes,j=i%256,group=j/32;
  uint scale,mn;
  if(group<4){scale=u8(off+4+group)&63;mn=u8(off+8+group)&63;}
  else {scale=(u8(off+8+group)&15)|((u8(off+group)>>6)<<4);mn=(u8(off+8+group)>>4)|((u8(off+4+group)>>6)<<4);}
  uint qoff=off+(type==12?16:48);uint q=(u8(qoff+(j/64)*32+j%32)>>(((j%64)/32)*4))&15;
  if(type==13)q|=((u8(off+16+j%32)>>group)&1)<<4;
  return f16(off)*float(scale*q)-f16(off+2)*float(mn);}
 return uintBitsToFloat(0x7fc00000u);
}
float sigmoid(float x){return 1.0/(1.0+exp(-x));}
float silu(float x){return x*sigmoid(x);}
float rounded_bf(float x){uint z=floatBitsToUint(x);return uintBitsToFloat((z+32767+((z>>16)&1))&0xffff0000u);}

// Ordinary commands latch nonfinite values at the producing store. Full
// min/max/norm statistics are sampled separately; the flag stays set until RESET.
void finite_value(float value){
 if(p.n!=0&&(isnan(value)||isinf(value))){
  atomicOr(dbits[p.n+3],0x3f800000u);
  atomicExchange(dbits[p.n],0x7fc00000u);atomicExchange(dbits[p.n+1],0x7fc00000u);atomicExchange(dbits[p.n+2],0x7fc00000u);
 }
}
