#pragma once
// Layout of the actual captured V11 scratch buffer.
struct Work {
  uint32_t end = 0;
  uint32_t take(uint32_t n) {
    uint32_t at = end;
    end += (n + 63) & ~63u;
    return at;
  }
  uint32_t R = take(10240), xn = take(10240), lo = take(320),
           gate = take(10240), inject = take(4), mixed = take(2560),
           block = take(2560), qkv = take(10240), h = take(10240),
           z = take(6144), alpha = take(48), beta = take(48), rec = take(6144),
           y = take(6144), qfull = take(12288), q = take(6144),
           kcur = take(512), vcur = take(512), idxraw = take(128),
           idxq = take(512), scores = take(2050), selected = take(2051),
           partial = take(24 * 65 * 258), attn = take(6144), shg = take(640),
           shu = take(640), shh = take(640), sho = take(2560), shgate = take(1),
           moe = take(2560), head = take(248320), pleemb = take(2560),
           plekey = take(10240), plequery = take(10240), pleval = take(2560),
           plegated = take(10240), plenorm = take(10240), round = take(12288),
           Rin = take(10240);
};
