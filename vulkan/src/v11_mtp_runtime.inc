// V11 speculative execution. Every command and descriptor is recorded once.
// Prefix snapshots omit KV: unwritten/rejected suffix cells are outside the
// committed logical position and are overwritten on the next verification.
void load_mtp() {
  auto &z = layers_[49];
  auto data = read_bytes((fs::path(a_.mtp) / "dense.bin").string());
  std::ifstream index(fs::path(a_.mtp) / "dense.txt");
  std::string name, kind;
  uint32_t rows, cols;
  uint64_t off, bytes;
  const std::map<std::string, std::string> aliases = {
      {"self_attn.q_proj.weight", "attn_q.weight"},
      {"self_attn.k_proj.weight", "attn_k.weight"},
      {"self_attn.v_proj.weight", "attn_v.weight"},
      {"self_attn.o_proj.weight", "attn_output.weight"},
      {"self_attn.q_norm.weight", "attn_q_norm.weight"},
      {"self_attn.k_norm.weight", "attn_k_norm.weight"},
      {"mlp.gate.weight", "ffn_gate_inp.weight"},
      {"mlp.shared_expert.gate_proj.weight", "ffn_gate_shexp.weight"},
      {"mlp.shared_expert.up_proj.weight", "ffn_up_shexp.weight"},
      {"mlp.shared_expert.down_proj.weight", "ffn_down_shexp.weight"},
      {"mlp.shared_expert_gate.weight", "ffn_gate_inp_shexp.weight"}};
  while (index >> name >> kind >> rows >> cols >> off >> bytes) {
    uint32_t type = kind == "q8_0"   ? 8
                    : kind == "bf16" ? 30
                    : kind == "f32"  ? 0
                                     : UINT32_MAX;
    uint64_t expected = uint64_t(rows) * cols *
                        (type == 0    ? 4
                         : type == 30 ? 2
                                      : 0);
    if (type == 8) {
      if (cols % 32)
        throw std::runtime_error("MTP Q8 alignment");
      expected = uint64_t(rows) * cols / 32 * 34;
    }
    if (type == UINT32_MAX || off + bytes > data.size() || bytes != expected ||
        off > UINT32_MAX)
      throw std::runtime_error("invalid MTP dense record " + name);
    std::string mapped = aliases.contains(name) ? aliases.at(name) : name;
    const std::pair<const char *, const char *> prefixes[] = {
        {"attn_hyper_connection.", "hc_attn_"},
        {"mlp_hyper_connection.", "hc_ffn_"},
        {"hyper_connection_mixer.", "output_hc_"}};
    for (auto [from, to] : prefixes)
      if (name.starts_with(from)) {
        std::string tail = name.substr(std::strlen(from));
        if (tail == "hc_norm.weight")
          mapped = std::string(to) + "norm.weight";
        else if (tail == "input_mix_weight_down.weight")
          mapped = std::string(to) + "down.weight";
        else if (tail == "input_mix_weight_up.weight")
          mapped = std::string(to) + "up.weight";
        else if (tail == "block_inject_weight.weight")
          mapped = std::string(to) + "inject.weight";
      }
    z.tensors["blk.49." + mapped] = {uint32_t(off), type, cols, rows,
                                     uint32_t(bytes / rows)};
  }
  if (!index.eof() || z.tensors.size() != 29)
    throw std::runtime_error("incomplete MTP dense index");
  data.resize(data.size() + 4);
  z.dense = upload_device(ctx_, data.data(), data.size());
  auto raw = read_bytes((fs::path(a_.mtp) / "experts.bin").string());
  constexpr uint64_t blob = 1382400;
  if (raw.size() != 512 * blob)
    throw std::runtime_error("invalid MTP expert payload");
  std::vector<uint8_t> native(raw.size());
  std::vector<uint32_t> meta(512 * 8);
  // Losslessly convert the existing planar/interleaved Q2_0 runtime pack
  // to V8's native block layout. No extra device weight copy is retained.
  for (uint32_t e = 0; e < 512; e++) {
    auto *src = raw.data() + e * blob;
    auto *dst = native.data() + e * blob;
    auto *m = meta.data() + e * 8;
    m[0] = 1;
    m[1] = m[2] = 42;
    m[3] = e * blob;
    m[4] = e * blob + 460800;
    m[5] = e * blob + 921600;
    m[6] = 720;
    m[7] = 180;
    for (uint32_t role = 0; role < 3; role++) {
      uint32_t nr = role < 2 ? 640 : 2560, nb = role < 2 ? 40 : 10;
      for (uint32_t row = 0; row < nr; row++)
        for (uint32_t b = 0; b < nb; b++) {
          uint32_t ir = role < 2 ? row * 2 + role : row;
          uint64_t co =
              role < 2 ? (ir * 40 + b) * 16 : 819200 + (ir * 10 + b) * 16;
          uint64_t sc = role < 2 ? 1228800 + (ir * 40 + b) * 2
                                 : 1331200 + (ir * 10 + b) * 2;
          auto *out =
              dst + (role < 2 ? role * 460800 : 921600) + (row * nb + b) * 18;
          std::memcpy(out, src + sc, 2);
          std::memcpy(out + 2, src + co, 16);
        }
    }
  }
  z.vram = upload_device(ctx_, native.data(), native.size());
  z.ram = buffer(4);
  z.plan = upload_device(ctx_, meta.data(), meta.size() * 4);
  z.gu_type = z.d_type = 42;
  z.state = buffer(4);
  z.kv = buffer(uint64_t(a_.context) * 512 * 2 * 2);
  auto vocab = read_bytes((fs::path(a_.mtp) / "draft_vocab.bin").string());
  if (vocab.empty() || vocab.size() % 4)
    throw std::runtime_error("invalid draft vocabulary");
  draft_vocab_count_ = vocab.size() / 4;
  std::vector<bool> seen(VOCAB);
  for (uint32_t i = 0; i < draft_vocab_count_; i++) {
    uint32_t id;
    std::memcpy(&id, vocab.data() + i * 4, 4);
    if (id >= VOCAB || seen[id])
      throw std::runtime_error("invalid/duplicate draft token");
    seen[id] = true;
  }
  draft_vocab_ = upload_device(ctx_, vocab.data(), vocab.size());
  if (!a_.forced.empty()) {
    auto ids = read_bytes(a_.forced);
    if (ids.size() % 4)
      throw std::runtime_error("forced ID alignment");
    forced_.resize(ids.size() / 4);
    std::memcpy(forced_.data(), ids.data(), ids.size());
  }
  std::cerr << "MTP weights: " << native.size() << " B experts, " << data.size()
            << " B dense; shared main embedding/head; draft vocabulary "
            << draft_vocab_count_ << "; spec " << a_.spec << " min-p "
            << a_.min_p << "\n";
}
template <class F> void each_row(uint32_t n, F f) {
  for (uint32_t row = 0; row < n; row++) {
    active_row_ = row;
    rounded_input_ = UINT32_MAX;
    f(row);
  }
  active_row_ = 0;
  rounded_input_ = UINT32_MAX;
}
void prepare_batch_input(uint32_t l, uint32_t in, uint32_t cols) {
  if (batch_input_ != in || batch_cols_ != cols) {
    Push qp{0, in, w_.round, cols};
    qp.l = w_.end;
    qp.n = finite_at();
    run(layers_[l].batch_qround, cols / 32, verify_rows_, &qp, sizeof(qp));
    rounded_input_ = UINT32_MAX;
    batch_input_ = in;
    batch_cols_ = cols;
  }
}
void batch_gemv(uint32_t l, const std::string &name, uint32_t in, uint32_t out,
                uint32_t cols, uint32_t rows, bool checked = false,
                bool independent = false) {
  auto &t = named(l, name, cols, rows);
  if (t.type != 0 && t.type != 1 && t.type != 30) {
    prepare_batch_input(l, in, cols);
    in = w_.round;
  }
  Push p{0, t.off, in, out, cols, rows, t.type, t.row_bytes};
  p.l = w_.end;
  p.o = verify_rows_;
  if (checked)
    p.n = finite_at(name == "output.weight");
  if ((t.type == 0 || t.type == 30 || t.type == 1) &&
      (rows <= 64 || cols >= 8192))
    run(layers_[l].batch_small[verify_rows_].at(t.type), rows, 1, &p, sizeof(p),
        !independent);
  else
    run(layers_[l].batch_dense[verify_rows_].at(t.type), (rows + 15) / 16, 1,
        &p, sizeof(p), !independent);
}
void batch_hc_read(uint32_t l, const std::string &prefix, bool apply = false) {
  auto &gamma = named(l, prefix + "norm.weight", HC);
  Push np{apply ? 25u : 1u, w_.R,     w_.xn, gamma.off, E, 4, 2, 0,
          w_.block,         w_.inject};
  np.l = w_.end;
  np.o = verify_rows_;
  batch_input_ = rounded_input_ = UINT32_MAX;
  run(layers_[l].batch_norm, 4, verify_rows_, &np, sizeof(np));
  auto &down = named(l, prefix + "down.weight", HC, 320);
  auto *inject = l < 48 ? &named(l, prefix + "inject.weight", HC, 4) : nullptr;
  Push dp{1,        down.off,
          w_.xn,    w_.lo,
          HC,       inject ? 324u : 320u,
          30,       HC * 2,
          0,        inject ? inject->off : 0,
          w_.inject};
  dp.l = w_.end;
  dp.o = verify_rows_;
  run(layers_[l].batch_hc_down[verify_rows_], dp.e, 1, &dp, sizeof(dp));
  auto &up = named(l, prefix + "up.weight", 320, HC);
  Push hp{0, up.off, w_.lo, w_.mixed, w_.xn};
  hp.l = w_.end;
  hp.o = verify_rows_;
  hp.f = w_.round;
  hp.n = finite_at();
  run(layers_[l].batch_hc[verify_rows_], E / 32, 1, &hp, sizeof(hp));
  batch_input_ = w_.mixed;
  batch_cols_ = E;
}
void transfer_barrier(bool to_transfer) {
  VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  b.dstAccessMask =
      to_transfer ? (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT)
                  : (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
  vkCmdPipelineBarrier(cmd_,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                       to_transfer ? VK_PIPELINE_STAGE_TRANSFER_BIT
                                   : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0, 1, &b, 0, nullptr, 0, nullptr);
}
void snapshot_tail(uint32_t l, uint32_t row, bool entire = false) {
  if (row == 4 || row + 1 == verify_rows_)
    return;
  uint64_t start = entire ? 0 : 128ull * 48 * 128 * 4;
  transfer_barrier(true);
  VkBufferCopy cp{start, row * snapshot_stride_ + snapshot_offset_[l] + start,
                  (entire ? snapshot_bytes_[l] - start : HC * 3ull * 4)};
  vkCmdCopyBuffer(cmd_, layers_[l].state.buffer, snapshots_.buffer, 1, &cp);
  transfer_barrier(false);
}
void batch_gdn(uint32_t l) {
  batch_gemv(l, "attn_qkv.weight", w_.mixed, w_.qkv, E, HC, false, true);
  batch_gemv(l, "ssm_alpha.weight", w_.mixed, w_.alpha, E, 48, false, true);
  batch_gemv(l, "ssm_beta.weight", w_.mixed, w_.beta, E, 48, false, true);
  batch_gemv(l, "attn_gate.weight", w_.mixed, w_.z, E, 6144);
  auto &cw = named(l, "ssm_conv1d.weight", 4, HC),
       &dt = named(l, "ssm_dt.bias", 48), &aa = named(l, "ssm_a", 48);
  Push prep{0,        w_.qkv,  w_.h,   128 * 48 * 128, cw.off,
            w_.alpha, w_.beta, dt.off, aa.off};
  prep.i = verify_rows_ == 1 ? 0 : snapshot_stride_ / 4;
  prep.j = snapshot_offset_[l] / 4;
  prep.l = w_.end;
  prep.o = verify_rows_;
  prep.n = finite_at();
  run(layers_[l].batch_gdn_prepare, 40, 1, &prep, sizeof(prep));
  batch_input_ = rounded_input_ = UINT32_MAX;
  Push p{0, w_.h, w_.rec, w_.alpha, w_.beta};
  p.i = verify_rows_ == 1 ? 0 : snapshot_stride_ / 4;
  p.j = snapshot_offset_[l] / 4;
  p.l = w_.end;
  p.o = verify_rows_;
  p.n = finite_at();
  run(layers_[l].batch_gdn, 32, 48, &p, sizeof(p));
  auto &gn = named(l, "ssm_norm.weight", 128);
  Push np{7, w_.rec, w_.y, gn.off, w_.z};
  np.l = w_.end;
  np.f = w_.round;
  np.n = finite_at();
  run(layers_[l].batch_gdn_output, 48, verify_rows_, &np, sizeof(np));
  rounded_input_ = UINT32_MAX;
  batch_input_ = w_.y;
  batch_cols_ = 6144;
  batch_gemv(l, "ssm_out.weight", w_.y, w_.block, 6144, E, true);
}
void qsa_body(uint32_t l, bool draft = false) {
  if (!draft) {
    auto &kn = named(l, "attn_k_norm.weight", 256);
    norm(l, w_.kcur, w_.kcur, 256, 2, &kn);
    ops(l, {9, w_.kcur, 256, 2});
    ops(l, {10, w_.kcur, w_.vcur, 0, a_.context});
  }
  if (!draft) {
    auto &ikn = named(l, "indexer.k_norm.weight", 128);
    ops(l, {11, w_.idxraw, 0, ikn.off, 512, 384});
  }
  ops(l, {8, w_.qfull, w_.q}, 24);
  auto &qn = named(l, "attn_q_norm.weight", 256);
  norm(l, w_.q, w_.q, 256, 24, &qn);
  ops(l, {9, w_.q, 256, 24}, 3);
  if (!draft) {
    auto &iqn = named(l, "indexer.q_norm.weight", 128);
    norm(l, w_.idxq, w_.idxq, 128, 4, &iqn);
    ops(l, {9, w_.idxq, 128, 4});
    ops(l, {12, w_.idxq, 512, 384, w_.scores}, a_.context / 4 + 1, 8);
    select_qsa(l);
  }
  Push p{0, w_.q, w_.partial, w_.selected, 0, a_.context};
  run(path(l).attn, 24, 65, &p, sizeof(p), true, 4);
  p = {1, w_.partial, w_.attn};
  run(path(l).attn, 24, 1, &p, sizeof(p));
  ops(l, {14, w_.attn, w_.qfull, w_.y}, 24);
}
void batch_qsa(uint32_t l) {
  prepare_batch_input(l, w_.mixed, E);
  batch_gemv(l, "indexer.k_proj.weight", w_.mixed, w_.idxraw, E, 128, false,
             true);
  batch_gemv(l, "attn_k.weight", w_.mixed, w_.kcur, E, 512, false, true);
  batch_gemv(l, "attn_v.weight", w_.mixed, w_.vcur, E, 512, false, true);
  batch_gemv(l, "attn_q.weight", w_.mixed, w_.qfull, E, 12288, false, true);
  batch_gemv(l, "indexer.q_proj.weight", w_.mixed, w_.idxq, E, 512);
  auto qops = [&](Push p, uint32_t groups) {
    p.l = w_.end;
    p.k = ctl_stride_ / 4;
    run(layers_[l].batch_qsa_ops, groups, verify_rows_, &p, sizeof(p));
    batch_input_ = rounded_input_ = UINT32_MAX;
  };
  auto prepare = [&](uint32_t op, uint32_t in, uint32_t out,
                     const Tensor &gamma, uint32_t width, uint32_t heads,
                     bool synchronize) {
    Push p{op, in, out, gamma.off, width, w_.vcur, a_.context};
    p.l = w_.end;
    p.k = ctl_stride_ / 4;
    run(layers_[l].batch_qsa_prepare, heads, verify_rows_, &p, sizeof(p),
        synchronize);
    batch_input_ = rounded_input_ = UINT32_MAX;
  };
  prepare(1, w_.kcur, w_.kcur, named(l, "attn_k_norm.weight", 256), 256, 2,
          false);
  prepare(0, w_.qfull, w_.q, named(l, "attn_q_norm.weight", 256), 256, 24,
          false);
  prepare(2, w_.idxq, w_.idxq, named(l, "indexer.q_norm.weight", 128), 128, 4,
          true);
  auto &ikn = named(l, "indexer.k_norm.weight", 128);
  each_row(verify_rows_, [&](uint32_t row) {
    ops(l, {11, w_.idxraw, 0, ikn.off, 512, 384});
    snapshot_tail(l, row, true);
  });
  // A later pooled-key append only modifies blocks at or above an earlier
  // position's nb. Earlier positions read rows strictly below nb and use the
  // fixed spare key for their tail, so all window scores can run together.
  Push score{12, w_.idxq, 512, 384, w_.scores};
  score.l = w_.end;
  score.k = ctl_stride_ / 4;
  run(layers_[l].batch_qsa_score, 1, 1, &score, sizeof(score), true,
      16 + 2 * E + 12);
  Push selection{0, w_.scores, 0, w_.selected};
  selection.l = w_.end;
  selection.k = ctl_stride_ / 4;
  run(layers_[l].batch_selection, 1, 1, &selection, sizeof(selection), true,
      16 + 2 * E + 8);
  Push ap{0, w_.q, w_.partial, w_.selected, 0, a_.context};
  ap.l = w_.end;
  ap.k = ctl_stride_ / 4;
  // Existing reserved control words select exactly one attention producer.
  // Keep the RC kernel for a whole window crossing the 512-position boundary.
  run(layers_[l].batch_attn, 1, 1, &ap, sizeof(ap), false, 16 + 2 * E);
  run(layers_[l].batch_attn_tile, 1, 1, &ap, sizeof(ap), true, 16 + 2 * E + 3);
  ap = {1, w_.partial, w_.attn};
  ap.l = w_.end;
  ap.k = ctl_stride_ / 4;
  run(layers_[l].batch_attn, 24, verify_rows_, &ap, sizeof(ap));
  qops({14, w_.attn, w_.qfull, w_.y}, 24);
  batch_gemv(l, "attn_output.weight", w_.y, w_.block, 6144, E, true);
}
uint32_t batch_gate_rows(uint32_t type, uint32_t n) {
  if (n == 2 && (type == 18 || type == 21))
    return 16;
  if (n == 3 || (n == 2 && type == 17))
    return 8;
  return 4;
}
const char *batch_expert_shader(uint32_t type, uint32_t rows = 4) {
  if (rows == 16) {
    if (type == 18)
      return JR_V11_EXPERT128_T18_SHADER;
    if (type == 21)
      return JR_V11_EXPERT128_T21_SHADER;
    throw std::runtime_error("unsupported sixteen-row verification format");
  }
  if (rows == 8) {
    switch (type) {
#define V11_WIDE_CASE(t)                                                       \
  case t:                                                                      \
    return JR_V11_EXPERT64_T##t##_SHADER;
      V11_WIDE_CASE(16)
      V11_WIDE_CASE(17)
      V11_WIDE_CASE(18)
      V11_WIDE_CASE(20)
      V11_WIDE_CASE(21)
      V11_WIDE_CASE(22)
      V11_WIDE_CASE(42)
#undef V11_WIDE_CASE
    default:
      throw std::runtime_error("unsupported eight-row verification format");
    }
  }
  switch (type) {
#define V11_EXPERT_CASE(t)                                                     \
  case t:                                                                      \
    return JR_V11_EXPERT_T##t##_SHADER;
    V11_EXPERT_CASE(16)
    V11_EXPERT_CASE(17)
    V11_EXPERT_CASE(18)
    V11_EXPERT_CASE(20)
    V11_EXPERT_CASE(21)
    V11_EXPERT_CASE(22)
    V11_EXPERT_CASE(42)
#undef V11_EXPERT_CASE
  default:
    throw std::runtime_error("unsupported V11 native batch expert type");
  }
}
void batch_expert(Pipeline &pipe, const std::array<uint32_t, 8> &push,
                  uint32_t indirect_words, bool synchronize = true) {
  vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
  vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.layout, 0,
                          1, &pipe.set, 0, nullptr);
  vkCmdPushConstants(cmd_, pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 32,
                     push.data());
  vkCmdDispatchIndirect(cmd_, group_slots_.buffer, indirect_words * 4);
  if (synchronize)
    barrier();
}
void batch_moe(uint32_t l) {
  batch_gemv(l, "ffn_gate_inp.weight", w_.mixed, w_.scores, E, 512, true);
  batch_mark(verify_rows_, "verify router/group");
  Push rp{0, w_.scores, 16 + l * 16};
  rp.l = w_.end;
  rp.k = route_stride_ / 4;
  rp.m = weights_stride_ / 4;
  rp.d = 65536 / 4;
  // Routing and Q8 packing are independent consumers of the ready mixed
  // input. One completion barrier covers both before grouped execution.
  run(layers_[l].batch_router, 1, verify_rows_, &rp, sizeof(rp), false);
  bool already_rounded = batch_input_ == w_.mixed && batch_cols_ == E;
  prepare_batch_input(l, w_.mixed, E);
  if (already_rounded)
    barrier();
  std::array<uint32_t, 4> group{
      verify_rows_, uint32_t(route_stride_ / 4),
      2 * FF / batch_gate_rows(layers_[l].gu_type, verify_rows_), E / 4};
  run(group_routes_, 1, 1, group.data(), 16, false);
  VkMemoryBarrier ready{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  ready.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  ready.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
  vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                       0, 1, &ready, 0, nullptr, 0, nullptr);
  // For N <= 4, prefix record 3 cannot be a restore point. Stream the
  // needed native host bytes through that existing workspace, then read them
  // with the unchanged expert decoder. N=5 retains all four live snapshots.
  if (verify_rows_ <= 4) {
    batch_mark(verify_rows_, "verify native prefetch");
    std::array<uint32_t, 2> cp{
        uint32_t(FF * row_bytes_for(layers_[l].gu_type, E)),
        uint32_t(E * row_bytes_for(layers_[l].d_type, FF))};
    auto &prefetch = layers_[l].verify_prefetch;
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, prefetch.pipeline);
    vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            prefetch.layout, 0, 1, &prefetch.set, 0, nullptr);
    vkCmdPushConstants(cmd_, prefetch.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8,
                       cp.data());
    vkCmdDispatchIndirect(cmd_, group_slots_.buffer, 508 * 4);
    barrier();
  }
  std::array<uint32_t, 8> gp{
      verify_rows_ * K,         E, FF, 0, verify_rows_, w_.end,
      uint32_t(gu_stride_ / 4), 0};
  batch_mark(verify_rows_, "verify expert GU/shared");
  batch_expert(layers_[l].batch_gu[verify_rows_], gp, 512, false);
  batch_gemv(l, "ffn_gate_shexp.weight", w_.mixed, w_.shg, E, FF, false, true);
  batch_gemv(l, "ffn_up_shexp.weight", w_.mixed, w_.shu, E, FF);
  batch_mark(verify_rows_, "verify SwiGLU");
  std::array<uint32_t, 8> sw{K,
                             FF,
                             uint32_t(gu_stride_ / 4),
                             uint32_t(hq_stride_ / 4),
                             w_.end,
                             w_.shg,
                             w_.shu,
                             w_.shh};
  run(batch_swiglu_, FF / 32, K * verify_rows_, sw.data(), 32);
  batch_input_ = rounded_input_ = UINT32_MAX;
  batch_mark(verify_rows_, "verify expert down/shared");
  prepare_batch_input(l, w_.shh, FF);
  gp[3] = 1;
  gp[5] = hq_stride_ / 4;
  gp[6] = parts_stride_ / 4;
  batch_expert(layers_[l].batch_down[verify_rows_], gp, 516, false);
  batch_gemv(l, "ffn_down_shexp.weight", w_.shh, w_.sho, FF, E, false, true);
  batch_gemv(l, "ffn_gate_inp_shexp.weight", w_.mixed, w_.shgate, E, 1);
  batch_mark(verify_rows_, "verify combine");
  std::array<uint32_t, 10> combine{K,
                                   E,
                                   uint32_t(parts_stride_ / 4),
                                   uint32_t(weights_stride_ / 4),
                                   w_.end,
                                   w_.moe,
                                   w_.sho,
                                   w_.shgate,
                                   w_.block,
                                   finite_at()};
  run(batch_combine_, 10, verify_rows_, combine.data(), 40);
  batch_input_ = rounded_input_ = UINT32_MAX;
}
VkCommandBuffer allocate_command() {
  VkCommandBufferAllocateInfo a{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  a.commandPool = pool_;
  a.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  a.commandBufferCount = 1;
  VkCommandBuffer c;
  vk_check(vkAllocateCommandBuffers(ctx_.device(), &a, &c),
           "allocate V11 persistent command");
  return c;
}
void begin_command(VkCommandBuffer c) {
  batch_input_ = UINT32_MAX;
  cmd_ = c;
  VkCommandBufferBeginInfo b{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  vk_check(vkBeginCommandBuffer(c, &b), "begin V11 command");
  VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  host.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                       VK_ACCESS_TRANSFER_WRITE_BIT;
  host.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                       VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
  vkCmdPipelineBarrier(c,
                       VK_PIPELINE_STAGE_HOST_BIT |
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                       0, 1, &host, 0, nullptr, 0, nullptr);
}
void end_command() {
  VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  b.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(
      cmd_,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
  vk_check(vkEndCommandBuffer(cmd_), "end V11 command");
}
void copy_scratch(uint32_t from_row, uint32_t from, uint32_t to_row,
                  uint32_t to, uint32_t n) {
  transfer_barrier(true);
  VkBufferCopy cp{(uint64_t(from_row) * w_.end + from) * 4,
                  (uint64_t(to_row) * w_.end + to) * 4, uint64_t(n) * 4};
  vkCmdCopyBuffer(cmd_, scratch_.buffer, scratch_.buffer, 1, &cp);
  transfer_barrier(false);
}
void draft_input() {
  uint32_t l = 49;
  active_row_ = 5;
  rounded_input_ = UINT32_MAX;
  Push ep{0, w_.xn, 0};
  ep.n = finite_at();
  run(embedding_[5], 40, 1, &ep, sizeof(ep));
  auto &en = named(l, "pre_fc_norm_embedding.weight", E);
  norm(l, w_.xn, w_.pleemb, E, 1, &en);
  gemv(l, "fc_embedding.weight", w_.pleemb, w_.pleval, E, E);
  auto &hn = named(l, "pre_fc_norm_hidden.weight", HC);
  norm(l, w_.Rin, w_.plenorm, HC, 1, &hn);
  for (uint32_t c = 0; c < 4; c++)
    gemv(l, "fc_hidden.weight", w_.plenorm + c * E, w_.R + c * E, E, E);
  ops(l, {26, w_.R, w_.pleval}, 40);
  hc_read(l, "hc_attn_");
  gemv(l, "attn_k.weight", w_.mixed, w_.kcur, E, 512);
  gemv(l, "attn_v.weight", w_.mixed, w_.vcur, E, 512);
  auto &kn = named(l, "attn_k_norm.weight", 256);
  norm(l, w_.kcur, w_.kcur, 256, 2, &kn);
  ops(l, {9, w_.kcur, 256, 2});
  ops(l, {10, w_.kcur, w_.vcur, 0, a_.context});
}
void draft_forward() {
  draft_input();
  gemv(49, "attn_q.weight", w_.mixed, w_.qfull, E, 12288);
  qsa_body(49, true);
  gemv(49, "attn_output.weight", w_.y, w_.block, 6144, E, false, true);
  hc_read(49, "hc_ffn_", true);
  moe(49);
  // The draft's final mixer is its own, while the native lm head is shared.
  auto &n = named(49, "output_hc_norm.weight", HC);
  ops(49, {25, w_.R, w_.xn, n.off, E, 4, 2, 0, w_.block, w_.inject}, 4);
  auto &down = named(49, "output_hc_down.weight", HC, 320);
  Push dp{1, down.off, w_.xn, w_.lo, HC, 320, 30, HC * 2};
  run(path(49).dense_small.at(30), 320, 1, &dp, sizeof(dp));
  auto &up = named(49, "output_hc_up.weight", 320, HC);
  Push hp{0, up.off, w_.lo, w_.mixed, w_.xn};
  run(path(49).hc, E / 4, 1, &hp, sizeof(hp));
  gemv(48, "output.weight", w_.mixed, w_.head, E, VOCAB, false, true, true);
  Push sp{22, w_.head, draft_vocab_count_};
  sp.m = 1;
  ops(48, sp);
  copy_scratch(5, w_.R, 5, w_.Rin, HC);
}
void batch_mark(uint32_t n, const std::string &name) {
  if (!a_.profile)
    return;
  batch_marks_[n].emplace_back(batch_nq_, name);
  vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      batch_queries_, batch_nq_++);
}
void copy_debug_logits(uint32_t row) {
  if (!head_host_.buffer)
    return;
  transfer_barrier(true);
  VkBufferCopy cp{(uint64_t(row) * w_.end + w_.head) * 4, 0, VOCAB * 4ull};
  vkCmdCopyBuffer(cmd_, scratch_.buffer, head_host_.buffer, 1, &cp);
}
void initialize_mtp_commands() {
  auto saved = cmd_;
  timing_enabled_ = false;
  if (a_.profile) {
    VkQueryPoolCreateInfo q{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    q.queryType = VK_QUERY_TYPE_TIMESTAMP;
    q.queryCount = 1024;
    vk_check(vkCreateQueryPool(ctx_.device(), &q, nullptr, &batch_queries_),
             "V11 profiling queries");
  }
  if (a_.draft_check) {
    draft_debug_ = buffer(w_.end * 8 + layers_[49].kv.size, true);
    draft_debug_cmd_ = allocate_command();
    begin_command(draft_debug_cmd_);
    transfer_barrier(true);
    VkBufferCopy a{5ull * w_.end * 4, 0, uint64_t(w_.end) * 4},
        b{0, uint64_t(w_.end) * 4, uint64_t(w_.end) * 4},
        k{0, uint64_t(w_.end) * 8, layers_[49].kv.size};
    vkCmdCopyBuffer(cmd_, scratch_.buffer, draft_debug_.buffer, 1, &a);
    vkCmdCopyBuffer(cmd_, scratch_.buffer, draft_debug_.buffer, 1, &b);
    vkCmdCopyBuffer(cmd_, layers_[49].kv.buffer, draft_debug_.buffer, 1, &k);
    end_command();
  }
  seed_cmd_ = allocate_command();
  begin_command(seed_cmd_);
  copy_scratch(0, w_.R, 5, w_.Rin, HC);
  end_command();
  draft_cmd_ = allocate_command();
  begin_command(draft_cmd_);
  draft_forward();
  end_command();
  for (uint32_t row = 0; row < 5; row++) {
    catchup_cmd_[row] = allocate_command();
    begin_command(catchup_cmd_[row]);
    copy_scratch(row, w_.R, 5, w_.Rin, HC);
    draft_input();
    end_command();
  }
  for (uint32_t n = 1; n <= 5; n++) {
    verify_rows_ = n;
    verify_cmd_[n] = allocate_command();
    begin_command(verify_cmd_[n]);
    batch_nq_ = 0;
    if (a_.profile)
      vkCmdResetQueryPool(cmd_, batch_queries_, 0, 1024);
    batch_mark(n, "embedding");
    each_row(n, [&](uint32_t) {
      Push ep{0, w_.R, 0};
      ep.n = finite_at();
      run(embedding_[active_row_], 40, 1, &ep, sizeof(ep),
          active_row_ + 1 == n);
    });
    for (uint32_t l = 0; l < 48; l++) {
      batch_mark(n, "PLE");
      if (l == 1)
        each_row(n, [&](uint32_t row) {
          ops(l, {4, w_.R, w_.block, w_.inject}, 40);
          ple();
          if (row == 4 || row + 1 == n)
            return;
          transfer_barrier(true);
          uint64_t start = (128ull * 48 * 128 + HC * 3) * 4;
          VkBufferCopy cp{start,
                          row * snapshot_stride_ + snapshot_offset_[l] + start,
                          HC * 9ull * 4};
          vkCmdCopyBuffer(cmd_, layers_[l].state.buffer, snapshots_.buffer, 1,
                          &cp);
          transfer_barrier(false);
        });
      batch_mark(n, "HC attention");
      batch_hc_read(l, "hc_attn_", l > 0 && l != 1);
      batch_mark(n, l % 4 == 3 ? "QSA" : "GDN");
      if (l % 4 == 3)
        batch_qsa(l);
      else
        batch_gdn(l);
      batch_mark(n, "HC FFN");
      batch_hc_read(l, "hc_ffn_", true);
      batch_mark(n, "MoE/shared");
      batch_moe(l);
    }
    batch_mark(n, "output HC");
    batch_hc_read(48, "output_hc_", true);
    batch_mark(n, "head");
    batch_gemv(48, "output.weight", w_.mixed, w_.head, E, VOCAB, true);
    each_row(n, [&](uint32_t) { ops(48, {22, w_.head, VOCAB}); });
    batch_mark(n, "end");
    end_command();
    restore_cmd_[n] = allocate_command();
    begin_command(restore_cmd_[n]);
    transfer_barrier(true);
    for (uint32_t l = 0; n < 5 && l < 48; l++) {
      VkBufferCopy cp{(n - 1) * snapshot_stride_ + snapshot_offset_[l], 0,
                      snapshot_bytes_[l]};
      vkCmdCopyBuffer(cmd_, snapshots_.buffer, layers_[l].state.buffer, 1, &cp);
    }
    transfer_barrier(false);
    if (n > 1)
      copy_scratch(n - 1, w_.R, 0, w_.R, HC);
    copy_debug_logits(n - 1);
    end_command();
    commit_cmd_[n] = allocate_command();
    begin_command(commit_cmd_[n]);
    if (n > 1)
      copy_scratch(n - 1, w_.R, 0, w_.R, HC);
    copy_debug_logits(n - 1);
    end_command();
  }
  active_row_ = 0;
  verify_rows_ = 0;
  cmd_ = saved;
  timing_enabled_ = true;
  rounded_input_ = UINT32_MAX;
  std::cerr << "V11 proposal/verification commands ready; prefix state storage "
            << snapshot_stride_ * 4 / (1 << 20) << " MiB\n";
  memory_report("MTP ready");
}
void stage_mtp(uint32_t cell, uint32_t token) {
  if (cell >= a_.context || token >= VOCAB)
    throw std::runtime_error("draft input out of range");
  auto *c = reinterpret_cast<uint32_t *>(static_cast<uint8_t *>(ctl_.mapped) +
                                         5 * ctl_stride_);
  c[0] = cell;
  c[1] = token;
  c[4] = cell >= 512 ? 3 : 24;
  c[5] = (cell + 1 + 31) / 32;
  c[6] = 1;
}
double execute(VkCommandBuffer c) {
  auto begin = std::chrono::steady_clock::now();
  vk_check(vkResetFences(ctx_.device(), 1, &fence_), "reset window fence");
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &c;
  vk_check(vkQueueSubmit(ctx_.queue(), 1, &si, fence_), "submit V11 window");
  vk_check(vkWaitForFences(ctx_.device(), 1, &fence_, VK_TRUE, 120000000000ull),
           "wait V11 window");
  for (uint32_t row = 0; row < rows_; row++) {
    auto *d = reinterpret_cast<float *>(static_cast<uint8_t *>(diag_.mapped) +
                                        row * 65536);
    if (d[1027] != 0 || !std::isfinite(d[1026])) {
      active_row_ = row;
      crash({1024, -1, "V11 proposal/verify"});
      throw std::runtime_error("nonfinite at position " +
                               std::to_string(position_));
    }
  }
  active_row_ = 0;
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - begin)
      .count();
}
void append_history(uint32_t id) {
  if (previous_count_ == 16) {
    std::memmove(previous_.data(), previous_.data() + 1, 60);
    previous_[15] = id;
  } else
    previous_[previous_count_++] = id;
  position_++;
}
std::string metrics_json() const {
  std::ostringstream o;
  o << std::setprecision(10) << "{\"proposal_hash\":\""
    << metrics_.proposal_hash << "\",\"peak_vram_gib\":" << peak_vram_
    << ",\"minimum_headroom_gib\":" << min_headroom_
    << ",\"proposed\":" << metrics_.proposed
    << ",\"accepted\":" << metrics_.accepted << ",\"acceptance_rate\":"
    << (metrics_.proposed ? double(metrics_.accepted) / metrics_.proposed : 0)
    << ",\"verification_windows\":" << metrics_.windows
    << ",\"mean_accepted_per_window\":"
    << (metrics_.windows ? double(metrics_.accepted) / metrics_.windows : 0)
    << ",\"mean_committed_window_length\":"
    << (metrics_.windows ? double(metrics_.committed) / metrics_.windows : 0)
    << ",\"committed\":" << metrics_.committed
    << ",\"draft_ms\":" << metrics_.draft_ms
    << ",\"draft_ms_per_proposed_token\":"
    << (metrics_.proposed ? metrics_.draft_ms / metrics_.proposed : 0)
    << ",\"verify_ms\":" << metrics_.verify_ms << ",\"verify_ms_per_window\":"
    << (metrics_.windows ? metrics_.verify_ms / metrics_.windows : 0)
    << ",\"rollback_count\":" << metrics_.rollbacks
    << ",\"rollback_ms\":" << metrics_.rollback_ms
    << ",\"catchup_ms\":" << metrics_.catchup_ms
    << ",\"total_ms\":" << metrics_.total_ms << ",\"effective_tok_s\":"
    << (metrics_.total_ms ? 1000 * metrics_.committed / metrics_.total_ms : 0);
  o << ",\"adaptive_max_drafts\":" << metrics_.adaptive_max_drafts;
  o << ",\"depth_proposed\":[";
  for (uint32_t k = 0; k < 4; k++)
    o << (k ? "," : "") << metrics_.depth_proposed[k];
  o << "],\"depth_accepted\":[";
  for (uint32_t k = 0; k < 4; k++)
    o << (k ? "," : "") << metrics_.depth_accepted[k];
  o << "],\"target_window_sizes\":[";
  for (uint32_t k = 0; k < 5; k++)
    o << (k ? "," : "") << metrics_.window_sizes[k];
  o << "]}";
  return o.str();
}
struct WindowResult {
  std::array<uint32_t, 5> tokens{};
  uint32_t count = 0;
};
WindowResult emit_window(const WindowResult &r, bool emit) {
  if (emit) {
    std::cout << "WINDOW {\"tokens\":[";
    for (uint32_t i = 0; i < r.count; i++)
      std::cout << (i ? "," : "") << r.tokens[i];
    std::cout << "],\"metrics\":" << metrics_json() << "}\n";
  }
  return r;
}
WindowResult round(uint32_t id, uint32_t limit, bool emit = true) {
  if (!limit || id >= VOCAB || position_ + limit > a_.context || !position_)
    throw std::runtime_error("invalid verification window");
  if (!a_.spec) {
    auto r = token(id);
    return emit_window({{r.token}, 1}, emit);
  }
  auto begin = std::chrono::steady_clock::now();
  uint32_t base = position_, saved_count = previous_count_;
  auto saved_history = previous_;
  std::array<uint32_t, 5> inputs{}, targets{};
  inputs[0] = id;
  uint32_t n = 1, draft_steps = 0;
  uint32_t budget = a_.spec;
  // Choose proposal depth from complete measured window cost, including
  // rejected rows. Confidence still bounds each candidate; occasional
  // maximum-depth probes allow the request to adapt when the text/context
  // changes.
  if (a_.adaptive_spec && a_.min_p > 0 && metrics_.windows >= 16 &&
      metrics_.windows % 64 != 0) {
    double best = 0;
    for (uint32_t d = 1; d <= a_.spec; d++)
      if (metrics_.window_sizes[d] >= 4 && metrics_.window_cost[d] > 0) {
        double rate = metrics_.window_commits[d] / metrics_.window_cost[d];
        if (rate > best) {
          best = rate;
          budget = d;
        }
      }
  }
  metrics_.adaptive_max_drafts = budget;
  metrics_.draft_ms += execute(seed_cmd_);
  for (uint32_t k = 0; k < std::min(budget, limit - 1); k++) {
    stage_mtp(base - 1 + k, inputs[k]);
    metrics_.draft_ms += execute(draft_cmd_);
    draft_steps++;
    if (a_.draft_check && !draft_dumped_) {
      execute(draft_debug_cmd_);
      fs::path dir = a_.dump.empty() ? "/tmp/v11-draft-check" : a_.dump;
      fs::create_directories(dir);
      write_f32((dir / "draft.f32").string(),
                static_cast<float *>(draft_debug_.mapped), w_.end);
      write_f32((dir / "base.f32").string(),
                static_cast<float *>(draft_debug_.mapped) + w_.end, w_.end);
      std::ofstream kvout(dir / "kv.bin", std::ios::binary);
      kvout.write(static_cast<char *>(draft_debug_.mapped) + w_.end * 8,
                  layers_[49].kv.size);
      std::ofstream meta(dir / "layout.json");
      meta << "{\"cell\":" << base - 1 << ",\"token\":" << inputs[k]
           << ",\"context\":" << a_.context;
#define V11_LAYOUT(x) meta << ",\"" #x "\":" << w_.x;
      V11_LAYOUT(R)
      V11_LAYOUT(plenorm)
      V11_LAYOUT(pleemb)
      V11_LAYOUT(pleval)
      V11_LAYOUT(qfull)
      V11_LAYOUT(q)
          V11_LAYOUT(kcur) V11_LAYOUT(vcur) V11_LAYOUT(attn) V11_LAYOUT(y)
              V11_LAYOUT(block) V11_LAYOUT(moe) V11_LAYOUT(scores)
                  V11_LAYOUT(shg) V11_LAYOUT(shu) V11_LAYOUT(sho)
                      V11_LAYOUT(shgate) V11_LAYOUT(mixed) V11_LAYOUT(inject)
#undef V11_LAYOUT
                          meta
          << "}\n";
      draft_dumped_ = true;
    }
    auto *d = reinterpret_cast<float *>(static_cast<uint8_t *>(diag_.mapped) +
                                        5 * 65536);
    uint32_t token = std::bit_cast<uint32_t>(d[0]);
    float prob = d[2];
    metrics_.proposal_hash =
        (metrics_.proposal_hash ^ token) * 1099511628211ull;
    metrics_.proposal_hash =
        (metrics_.proposal_hash ^ std::bit_cast<uint32_t>(prob)) *
        1099511628211ull;
    if (token >= VOCAB || !std::isfinite(prob) || prob < 0 || prob > 1)
      throw std::runtime_error("invalid MTP proposal");
    // Match the canonical MTP confidence policy: each proposed token must
    // meet the probability floor, rather than multiplying across the chain.
    if (prob < a_.min_p)
      break;
    inputs[n++] = token;
  }
  for (uint32_t k = 1; k < n; k++) {
    if (k <= test_override_count_)
      inputs[k] = test_override_[k - 1];
    if (!forced_.empty()) {
      if (base + k >= forced_.size())
        throw std::runtime_error("forced draft fixture exhausted");
      inputs[k] = forced_[base + k];
    }
    if (a_.force_reject == int(k - 1))
      inputs[k] = (inputs[k] + 1) % VOCAB;
  }
  test_override_count_ = 0;
  metrics_.proposed += n - 1;
  metrics_.window_sizes[n - 1]++;
  for (uint32_t k = 0; k + 1 < n; k++)
    metrics_.depth_proposed[k]++;
  // A terminal single-token window has not executed a draft input at all.
  // Preserve the ordinary paired submission so its preceding MTP KV is ready
  // if a caller continues the committed request later.
  if (n == 1 && draft_steps == 0) {
    auto r = token(id);
    metrics_.verify_ms += r.total_ms;
    metrics_.windows++;
    metrics_.committed++;
    metrics_.total_ms += std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - begin)
                             .count();
    return emit_window({{r.token}, 1}, emit);
  }
  // A zero-proposal window still verifies one target position. Reuse the
  // recorded batch command; no rejected suffix exists, so it needs no snapshot.

  each_row(n, [&](uint32_t row) {
    stage(inputs[row]);
    append_history(inputs[row]);
  });
  auto *dispatch = static_cast<uint32_t *>(ctl_.mapped) + 16 + 2 * E;
  dispatch[0] = base >= 512 ? 0 : 24;
  dispatch[3] = base >= 512 ? 2 : 0;
  dispatch[4] = (std::min(base + n, 2051u) + 31) / 32;
  dispatch[5] = n;
  dispatch[1] = (std::min(base + n, 2051u) + 31) / 32;
  dispatch[2] = n;
  uint32_t last_count = base + n;
  dispatch[8] =
      last_count <= 2051 ? (last_count + 255) / 256 : (last_count / 4 + 8) / 8;
  dispatch[9] = n;
  dispatch[10] = 1;
  dispatch[12] = last_count > 2051 ? last_count / 4 + 1 : 0;
  dispatch[13] = n;
  dispatch[14] = 1;
  position_ = base;
  previous_count_ = saved_count;
  previous_ = saved_history;
  metrics_.verify_ms += execute(verify_cmd_[n]);
  metrics_.windows++;
  if (a_.profile && metrics_.windows % 32 == 1) {
    auto &marks = batch_marks_[n];
    std::vector<uint64_t> stamps(marks.size());
    vk_check(vkGetQueryPoolResults(ctx_.device(), batch_queries_, 0,
                                   marks.size(), stamps.size() * 8,
                                   stamps.data(), 8, VK_QUERY_RESULT_64_BIT),
             "V11 batch profiling");
    std::map<std::string, double> costs;
    for (size_t i = 0; i + 1 < marks.size(); i++)
      costs[marks[i].second] +=
          double(stamps[i + 1] - stamps[i]) * timestamp_period_ * 1e-6;
    for (auto &[name, ms] : costs)
      std::cerr << "BATCH_PROFILE T=" << n << " " << name << " " << ms
                << " ms\n";
  }

  for (uint32_t row = 0; row < n; row++) {
    auto *d = reinterpret_cast<float *>(static_cast<uint8_t *>(diag_.mapped) +
                                        row * 65536);
    targets[row] = std::bit_cast<uint32_t>(d[0]);
    if (targets[row] >= VOCAB || !std::isfinite(d[1]))
      throw std::runtime_error("invalid verification sampler");
  }
  uint32_t accepted = 0;
  while (accepted + 1 < n && inputs[accepted + 1] == targets[accepted])
    accepted++;
  uint32_t committed = accepted + 1;
  if (!a_.ignore_eos)
    for (uint32_t row = 0; row < committed; row++)
      if (targets[row] == 248044 || targets[row] == 248046) {
        committed = row + 1;
        accepted = std::min(accepted, row);
        break;
      }
  metrics_.accepted += accepted;
  metrics_.committed += committed;
  for (uint32_t k = 0; k < accepted; k++)
    metrics_.depth_accepted[k]++;
  // Rebuild only the draft KV for committed target residual/token pairs. The
  // base verifier is never replayed; its exact chosen snapshot is restored.
  // The next proposal seeds the final committed pair itself. Refresh only
  // earlier pairs here, avoiding duplicate dense input/KV work.
  for (uint32_t row = 0; row + 1 < committed; row++) {
    stage_mtp(base + row, targets[row]);
    metrics_.catchup_ms += execute(catchup_cmd_[row]);
  }
  if (committed < n) {
    metrics_.rollbacks++;
    metrics_.rollback_ms += execute(restore_cmd_[committed]);
  } else if (n > 1 || head_host_.buffer) { // Commit the residual/debug head.
    metrics_.rollback_ms += execute(commit_cmd_[committed]);
  }
  position_ = base;
  previous_count_ = saved_count;
  previous_ = saved_history;
  for (uint32_t row = 0; row < committed; row++)
    append_history(inputs[row]);
  // Count routing only for committed rows, never rejected suffixes.
  for (uint32_t row = 0; row < committed; row++) {
    auto *d = reinterpret_cast<float *>(static_cast<uint8_t *>(diag_.mapped) +
                                        row * 65536);
    for (uint32_t l = 0; l < 48; l++)
      for (uint32_t k = 0; k < K; k++) {
        uint32_t e = std::bit_cast<uint32_t>(d[16 + l * 16 + k]);
        if (e >= 512)
          throw std::runtime_error("verify router bounds");
        route_counts_[l * 512 + e]++;
        route_hits_ += plan_.slot[l * 512 + e] >= 0;
        route_count_++;
      }
  }
  if (position_ / a_.stats_every != base / a_.stats_every) {
    memory_report("MTP position " + std::to_string(position_));
    std::cerr << "MTP metrics " << metrics_json() << "\n";
  }
  double elapsed = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - begin)
                       .count();
  metrics_.total_ms += elapsed;
  auto &cost = metrics_.window_cost[n - 1],
       &gain = metrics_.window_commits[n - 1];
  double decay = cost > 0 ? .9 : 0;
  cost = decay * cost + (1 - decay) * elapsed;
  gain = decay * gain + (1 - decay) * committed;
  if (head_host_.buffer)
    write_f32(a_.logits, static_cast<float *>(head_host_.mapped), VOCAB);
  return emit_window({targets, committed}, emit);
}
