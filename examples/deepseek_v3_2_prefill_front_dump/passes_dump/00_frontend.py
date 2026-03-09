# pypto.program: DeepSeekV32PrefillFront
import pypto.language as pl

@pl.program
class DeepSeekV32PrefillFront:
    @pl.function
    def deepseek_v3_2_prefill_front_layer(self, hidden_states: pl.Tensor[[16, 4096, 7168], pl.BFLOAT16], seq_lens: pl.Tensor[[16], pl.INT32], layer_id_t: pl.Tensor[[1], pl.INT32], rope_cos: pl.Tensor[[4096, 64], pl.FP32], rope_sin: pl.Tensor[[4096, 64], pl.FP32], kv_cache: pl.Tensor[[65536, 512], pl.BFLOAT16], pe_cache: pl.Tensor[[65536, 64], pl.BFLOAT16], input_rms_weight: pl.Tensor[[1, 7168], pl.FP32], wq_a: pl.Tensor[[7168, 1536], pl.BFLOAT16], q_norm_weight: pl.Tensor[[1, 1536], pl.FP32], wq_b: pl.Tensor[[1536, 24576], pl.BFLOAT16], wkv_a: pl.Tensor[[7168, 576], pl.BFLOAT16], kv_norm_weight: pl.Tensor[[1, 512], pl.FP32], w_q_nope_to_latent: pl.Tensor[[128, 128, 512], pl.BFLOAT16], w_latent_to_v: pl.Tensor[[128, 512, 128], pl.BFLOAT16], dispatch_buf: pl.Tensor[[128, 16, 4096, 16384], pl.BFLOAT16]) -> pl.Tensor[[128, 16, 4096, 16384], pl.BFLOAT16]:
        layer_id: pl.Scalar[pl.INT32] = pl.tensor.read(layer_id_t, [0])
        for b in pl.parallel(0, 16, 1, chunk=4):
            seq_len_b: pl.Scalar[pl.INT32] = pl.tensor.read(seq_lens, [b])
            tok_blocks: pl.Scalar[pl.INDEX] = (pl.cast(seq_len_b, pl.INDEX) + 4 - 1) // 4
            for p0_idx in pl.range(0, tok_blocks, 1):
                p0: pl.Scalar[pl.INDEX] = p0_idx * 4
                valid_tok: pl.Scalar[pl.INDEX] = min(4, pl.cast(seq_len_b, pl.INDEX) - p0)
                with pl.auto_incore():
                    sq_sum: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.create([4, 1], dtype=pl.FP32)
                    sq_sum: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.mul(sq_sum, 0.0)
                    usage_pad: pl.Tensor[[4, 16384], pl.BFLOAT16] = pl.tensor.create([4, 16384], dtype=pl.BFLOAT16)
                    usage_pad: pl.Tensor[[4, 16384], pl.FP32] = pl.tensor.mul(usage_pad, 0.0)
                    usage_pad_fp: pl.Tensor[[4, 16384], pl.FP32] = pl.tensor.cast(usage_pad, target_type=pl.FP32, mode=2)
                    usage_pad_sum: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.row_sum(usage_pad_fp)
                    for kb in pl.range(0, 14, 1):
                        k0: pl.Scalar[pl.INDEX] = kb * 512
                        x_chunk: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.cast(pl.tensor.view(hidden_states, [4, 512], [b, p0, k0]), target_type=pl.FP32, mode=2)
                        sq_sum: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.add(sq_sum, pl.tensor.row_sum(pl.tensor.mul(x_chunk, x_chunk)))
                    inv_rms: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.rsqrt(pl.tensor.add(pl.tensor.mul(sq_sum, 0.000139509), 1e-06))
                    inv_rms: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.add(inv_rms, pl.tensor.mul(usage_pad_sum, 0.0))
                    q_proj_tile: pl.Tensor[[4, 128 * 192], pl.BFLOAT16] = pl.tensor.create([4, 128 * 192], dtype=pl.BFLOAT16)
                    kv_a_tile: pl.Tensor[[4, 576], pl.BFLOAT16] = pl.tensor.create([4, 576], dtype=pl.BFLOAT16)
                    for ob in pl.parallel(0, 96, 1, chunk=8):
                        q0: pl.Scalar[pl.INDEX] = ob * 256
                        q_acc: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.create([4, 256], dtype=pl.FP32)
                        q_acc: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.mul(q_acc, 0.0)
                        for kb in pl.range(0, 14, 1):
                            k0: pl.Scalar[pl.INDEX] = kb * 512
                            x_chunk: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.cast(pl.tensor.view(hidden_states, [4, 512], [b, p0, k0]), target_type=pl.FP32, mode=2)
                            gamma_in: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.view(input_rms_weight, [1, 512], [0, k0])
                            normed: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(x_chunk, inv_rms), gamma_in)
                            for rb in pl.range(0, 12, 1):
                                r0: pl.Scalar[pl.INDEX] = rb * 128
                                wq_a_chunk: pl.Tensor[[512, 128], pl.BFLOAT16] = pl.tensor.view(wq_a, [512, 128], [k0, r0])
                                qr_part: pl.Tensor[[4, 128], pl.BFLOAT16] = pl.tensor.matmul(pl.tensor.cast(normed, target_type=pl.BFLOAT16, mode=2), wq_a_chunk, a_trans=False, b_trans=False, c_matrix_nz=False)
                                gamma_q: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.view(q_norm_weight, [1, 128], [0, r0])
                                qn_part: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.col_expand_mul(qr_part, gamma_q)
                                wq_b_chunk: pl.Tensor[[128, 256], pl.BFLOAT16] = pl.tensor.view(wq_b, [128, 256], [r0, q0])
                                q_acc: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.add(q_acc, pl.tensor.matmul(pl.tensor.cast(qn_part, target_type=pl.BFLOAT16, mode=2), wq_b_chunk, a_trans=False, b_trans=False, c_matrix_nz=False))
                        q_proj_tile: pl.Tensor[[4, 128 * 192], pl.BFLOAT16] = pl.tensor.assemble(q_proj_tile, pl.tensor.cast(q_acc, target_type=pl.BFLOAT16, mode=2), [0, q0])
                    for ob in pl.parallel(0, 5, 1, chunk=8):
                        kv0: pl.Scalar[pl.INDEX] = ob * 128
                        kv_acc: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.create([4, 128], dtype=pl.FP32)
                        kv_acc: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.mul(kv_acc, 0.0)
                        for kb in pl.range(0, 14, 1):
                            k0: pl.Scalar[pl.INDEX] = kb * 512
                            x_chunk: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.cast(pl.tensor.view(hidden_states, [4, 512], [b, p0, k0]), target_type=pl.FP32, mode=2)
                            gamma: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.view(input_rms_weight, [1, 512], [0, k0])
                            normed: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(x_chunk, inv_rms), gamma)
                            wkv_chunk: pl.Tensor[[512, 128], pl.BFLOAT16] = pl.tensor.view(wkv_a, [512, 128], [k0, kv0])
                            kv_acc: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.add(kv_acc, pl.tensor.matmul(pl.tensor.cast(normed, target_type=pl.BFLOAT16, mode=2), wkv_chunk, a_trans=False, b_trans=False, c_matrix_nz=False))
                        kv_a_tile: pl.Tensor[[4, 576], pl.BFLOAT16] = pl.tensor.assemble(kv_a_tile, pl.tensor.cast(kv_acc, target_type=pl.BFLOAT16, mode=2), [0, kv0])
                with pl.auto_incore():
                    attn_tile: pl.Tensor[[4, 16384], pl.FP32] = pl.tensor.create([4, 16384], dtype=pl.FP32)
                    attn_tile: pl.Tensor[[4, 16384], pl.FP32] = pl.tensor.mul(attn_tile, 0.0)
                    for ti in pl.range(0, valid_tok, 1):
                        pos: pl.Scalar[pl.INDEX] = p0 + ti
                        ctx_len: pl.Scalar[pl.INDEX] = pos + 1
                        cos_row: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.view(rope_cos, [1, 64], [pos, 0])
                        sin_row: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.view(rope_sin, [1, 64], [pos, 0])
                        cos_lo: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.view(cos_row, [1, 64 // 2], [0, 0])
                        cos_hi: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.view(cos_row, [1, 64 // 2], [0, 64 // 2])
                        sin_lo: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.view(sin_row, [1, 64 // 2], [0, 0])
                        sin_hi: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.view(sin_row, [1, 64 // 2], [0, 64 // 2])
                        cache_row: pl.Scalar[pl.INDEX] = b * 4096 + pos
                        kv_row: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.cast(pl.tensor.view(kv_a_tile, [1, 512], [ti, 0]), target_type=pl.FP32, mode=2)
                        kv_gamma: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.view(kv_norm_weight, [1, 512], [0, 0])
                        kv_normed: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.col_expand_mul(kv_row, kv_gamma)
                        pe_row: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(pl.tensor.view(kv_a_tile, [1, 64], [ti, 512]), target_type=pl.FP32, mode=2)
                        pe_lo: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.view(pe_row, [1, 64 // 2], [0, 0])
                        pe_hi: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.view(pe_row, [1, 64 // 2], [0, 64 // 2])
                        pe_rot: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.create([1, 64], dtype=pl.FP32)
                        pe_rot: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.assemble(pe_rot, pl.tensor.sub(pl.tensor.col_expand_mul(pe_lo, cos_lo), pl.tensor.col_expand_mul(pe_hi, sin_lo)), [0, 0])
                        pe_rot: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.assemble(pe_rot, pl.tensor.add(pl.tensor.col_expand_mul(pe_hi, cos_hi), pl.tensor.col_expand_mul(pe_lo, sin_hi)), [0, 64 // 2])
                        kv_cache: pl.Tensor[[65536, 512], pl.BFLOAT16] = pl.tensor.assemble(kv_cache, pl.tensor.cast(kv_normed, target_type=pl.BFLOAT16, mode=2), [cache_row, 0])
                        pe_cache: pl.Tensor[[65536, 64], pl.BFLOAT16] = pl.tensor.assemble(pe_cache, pl.tensor.cast(pe_rot, target_type=pl.BFLOAT16, mode=2), [cache_row, 0])
                        topk_vals: pl.Tensor[[1, 2048], pl.FP32] = pl.tensor.create([1, 2048], dtype=pl.FP32)
                        topk_idx: pl.Tensor[[1, 2048], pl.INT32] = pl.tensor.create([1, 2048], dtype=pl.INT32)
                        blk_topk_vals: pl.Tensor[[2, 2048], pl.FP32] = pl.tensor.create([2, 2048], dtype=pl.FP32)
                        blk_topk_idx: pl.Tensor[[2, 2048], pl.INT32] = pl.tensor.create([2, 2048], dtype=pl.INT32)
                        topk_vals: pl.Tensor[[1, 2048], pl.FP32] = pl.tensor.mul(topk_vals, -340282299999999994960115009090224128000.0)
                        topk_idx: pl.Tensor[[1, 2048], pl.INDEX] = pl.tensor.mul(topk_idx, 0)
                        blk_topk_vals: pl.Tensor[[2, 2048], pl.FP32] = pl.tensor.mul(blk_topk_vals, -340282299999999994960115009090224128000.0)
                        blk_topk_idx: pl.Tensor[[2, 2048], pl.INDEX] = pl.tensor.mul(blk_topk_idx, 0)
                        for kk in pl.range(0, 2048, 1):
                            neg_one: pl.Tensor[[1, 1], pl.INT32] = pl.tensor.create([1, 1], dtype=pl.INT32)
                            neg_one: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.mul(neg_one, 0)
                            neg_one: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.add(neg_one, -1)
                            topk_idx: pl.Tensor[[1, 2048], pl.INDEX] = pl.tensor.assemble(topk_idx, neg_one, [0, kk])
                            blk_topk_idx: pl.Tensor[[2, 2048], pl.INDEX] = pl.tensor.assemble(blk_topk_idx, neg_one, [0, kk])
                            blk_topk_idx: pl.Tensor[[2, 2048], pl.INDEX] = pl.tensor.assemble(blk_topk_idx, neg_one, [1, kk])
                        q_col0: pl.Scalar[pl.INDEX] = 0
                        q_nope0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.cast(pl.tensor.view(q_proj_tile, [1, 128], [ti, q_col0]), target_type=pl.FP32, mode=2)
                        q_pe0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(pl.tensor.view(q_proj_tile, [1, 64], [ti, q_col0 + 128]), target_type=pl.FP32, mode=2)
                        q0_lo: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.view(q_pe0, [1, 64 // 2], [0, 0])
                        q0_hi: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.view(q_pe0, [1, 64 // 2], [0, 64 // 2])
                        q0_rot: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.create([1, 64], dtype=pl.FP32)
                        q0_rot: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.assemble(q0_rot, pl.tensor.sub(pl.tensor.col_expand_mul(q0_lo, cos_lo), pl.tensor.col_expand_mul(q0_hi, sin_lo)), [0, 0])
                        q0_rot: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.assemble(q0_rot, pl.tensor.add(pl.tensor.col_expand_mul(q0_hi, cos_hi), pl.tensor.col_expand_mul(q0_lo, sin_hi)), [0, 64 // 2])
                        q0_nope_latent: pl.Tensor[[1, 512], pl.BFLOAT16] = pl.tensor.matmul(pl.tensor.cast(q_nope0, target_type=pl.BFLOAT16, mode=2), pl.tensor.view(w_q_nope_to_latent, [128, 512], [0, 0, 0]), a_trans=False, b_trans=False, c_matrix_nz=False)
                        sparse_k_gen: pl.Scalar[pl.INDEX] = min(2048, ctx_len)
                        for blk in pl.range(0, 2, 1):
                            blk_start: pl.Scalar[pl.INDEX] = blk * 2048
                            blk_end: pl.Scalar[pl.INDEX] = min(ctx_len, blk_start + 2048)
                            for ss in pl.range(0, 2048, 1):
                                s: pl.Scalar[pl.INDEX] = blk_start + ss
                                if s < blk_end:
                                    cache_s: pl.Scalar[pl.INDEX] = b * 4096 + s
                                    kv_s: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.cast(pl.tensor.view(kv_cache, [1, 512], [cache_s, 0]), target_type=pl.FP32, mode=2)
                                    pe_s: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(pl.tensor.view(pe_cache, [1, 64], [cache_s, 0]), target_type=pl.FP32, mode=2)
                                    score_nope: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(q0_nope_latent, kv_s))
                                    score_pe: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(q0_rot, pe_s))
                                    score_fp32: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(pl.tensor.add(score_nope, score_pe), 0.0721688)
                                    score_fp8: pl.Tensor[[1, 1], pl.FP8E4M3FN] = pl.tensor.cast(score_fp32, target_type=pl.FP8E4M3FN, mode=2)
                                    score_a5: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.cast(score_fp8, target_type=pl.FP32, mode=2)
                                    cur_score: pl.Scalar[pl.FP32] = pl.tensor.read(score_a5, [0, 0])
                                    inserted: pl.Tensor[[1, 1], pl.INT32] = pl.tensor.create([1, 1], dtype=pl.INT32)
                                    inserted: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.mul(inserted, 0)
                                    for kk in pl.range(0, sparse_k_gen, 1):
                                        ins: pl.Scalar[pl.INDEX] = pl.tensor.read(inserted, [0, 0])
                                        kth_val: pl.Scalar[pl.FP32] = pl.tensor.read(blk_topk_vals, [blk, kk])
                                        if ins == 0:
                                            if cur_score > kth_val:
                                                for sh in pl.range(sparse_k_gen - 1, kk, -1):
                                                    prev_val: pl.Scalar[pl.FP32] = pl.tensor.read(blk_topk_vals, [blk, sh - 1])
                                                    prev_idx: pl.Scalar[pl.INDEX] = pl.tensor.read(blk_topk_idx, [blk, sh - 1])
                                                    prev_val_t: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32)
                                                    prev_idx_t: pl.Tensor[[1, 1], pl.INT32] = pl.tensor.create([1, 1], dtype=pl.INT32)
                                                    prev_val_t: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(prev_val_t, 0.0)
                                                    prev_idx_t: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.mul(prev_idx_t, 0)
                                                    prev_val_t: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.add(prev_val_t, prev_val)
                                                    prev_idx_t: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.add(prev_idx_t, prev_idx)
                                                    blk_topk_vals: pl.Tensor[[2, 2048], pl.FP32] = pl.tensor.assemble(blk_topk_vals, prev_val_t, [blk, sh])
                                                    blk_topk_idx: pl.Tensor[[2, 2048], pl.INDEX] = pl.tensor.assemble(blk_topk_idx, prev_idx_t, [blk, sh])
                                                cur_score_t: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32)
                                                cur_index_t: pl.Tensor[[1, 1], pl.INT32] = pl.tensor.create([1, 1], dtype=pl.INT32)
                                                one_t: pl.Tensor[[1, 1], pl.INT32] = pl.tensor.create([1, 1], dtype=pl.INT32)
                                                cur_score_t: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(cur_score_t, 0.0)
                                                cur_index_t: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.mul(cur_index_t, 0)
                                                one_t: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.mul(one_t, 0)
                                                cur_score_t: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.add(cur_score_t, cur_score)
                                                cur_index_t: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.add(cur_index_t, s)
                                                one_t: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.add(one_t, 1)
                                                blk_topk_vals: pl.Tensor[[2, 2048], pl.FP32] = pl.tensor.assemble(blk_topk_vals, cur_score_t, [blk, kk])
                                                blk_topk_idx: pl.Tensor[[2, 2048], pl.INDEX] = pl.tensor.assemble(blk_topk_idx, cur_index_t, [blk, kk])
                                                inserted: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.assemble(inserted, one_t, [0, 0])
                        for blk in pl.range(0, 2, 1):
                            for kk in pl.range(0, sparse_k_gen, 1):
                                cand_idx: pl.Scalar[pl.INDEX] = pl.tensor.read(blk_topk_idx, [blk, kk])
                                if cand_idx >= 0:
                                    cand_val: pl.Scalar[pl.FP32] = pl.tensor.read(blk_topk_vals, [blk, kk])
                                    inserted: pl.Tensor[[1, 1], pl.INT32] = pl.tensor.create([1, 1], dtype=pl.INT32)
                                    inserted: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.mul(inserted, 0)
                                    for tkk in pl.range(0, sparse_k_gen, 1):
                                        ins: pl.Scalar[pl.INDEX] = pl.tensor.read(inserted, [0, 0])
                                        kth_val: pl.Scalar[pl.FP32] = pl.tensor.read(topk_vals, [0, tkk])
                                        if ins == 0:
                                            if cand_val > kth_val:
                                                for sh in pl.range(sparse_k_gen - 1, tkk, -1):
                                                    prev_val: pl.Scalar[pl.FP32] = pl.tensor.read(topk_vals, [0, sh - 1])
                                                    prev_idx: pl.Scalar[pl.INDEX] = pl.tensor.read(topk_idx, [0, sh - 1])
                                                    prev_val_t: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32)
                                                    prev_idx_t: pl.Tensor[[1, 1], pl.INT32] = pl.tensor.create([1, 1], dtype=pl.INT32)
                                                    prev_val_t: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(prev_val_t, 0.0)
                                                    prev_idx_t: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.mul(prev_idx_t, 0)
                                                    prev_val_t: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.add(prev_val_t, prev_val)
                                                    prev_idx_t: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.add(prev_idx_t, prev_idx)
                                                    topk_vals: pl.Tensor[[1, 2048], pl.FP32] = pl.tensor.assemble(topk_vals, prev_val_t, [0, sh])
                                                    topk_idx: pl.Tensor[[1, 2048], pl.INDEX] = pl.tensor.assemble(topk_idx, prev_idx_t, [0, sh])
                                                cand_val_t: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32)
                                                cand_idx_t: pl.Tensor[[1, 1], pl.INT32] = pl.tensor.create([1, 1], dtype=pl.INT32)
                                                one_t: pl.Tensor[[1, 1], pl.INT32] = pl.tensor.create([1, 1], dtype=pl.INT32)
                                                cand_val_t: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(cand_val_t, 0.0)
                                                cand_idx_t: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.mul(cand_idx_t, 0)
                                                one_t: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.mul(one_t, 0)
                                                cand_val_t: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.add(cand_val_t, cand_val)
                                                cand_idx_t: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.add(cand_idx_t, cand_idx)
                                                one_t: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.add(one_t, 1)
                                                topk_vals: pl.Tensor[[1, 2048], pl.FP32] = pl.tensor.assemble(topk_vals, cand_val_t, [0, tkk])
                                                topk_idx: pl.Tensor[[1, 2048], pl.INDEX] = pl.tensor.assemble(topk_idx, cand_idx_t, [0, tkk])
                                                inserted: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.assemble(inserted, one_t, [0, 0])
                        attn_row: pl.Tensor[[1, 16384], pl.FP32] = pl.tensor.create([1, 16384], dtype=pl.FP32)
                        attn_row: pl.Tensor[[1, 16384], pl.FP32] = pl.tensor.mul(attn_row, 0.0)
                        for h in pl.parallel(0, 128, 1, chunk=8):
                            q_col: pl.Scalar[pl.INDEX] = h * 192
                            q_nope: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.cast(pl.tensor.view(q_proj_tile, [1, 128], [ti, q_col]), target_type=pl.FP32, mode=2)
                            q_pe: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(pl.tensor.view(q_proj_tile, [1, 64], [ti, q_col + 128]), target_type=pl.FP32, mode=2)
                            q_lo: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.view(q_pe, [1, 64 // 2], [0, 0])
                            q_hi: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.view(q_pe, [1, 64 // 2], [0, 64 // 2])
                            q_rot: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.create([1, 64], dtype=pl.FP32)
                            q_rot: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.assemble(q_rot, pl.tensor.sub(pl.tensor.col_expand_mul(q_lo, cos_lo), pl.tensor.col_expand_mul(q_hi, sin_lo)), [0, 0])
                            q_rot: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.assemble(q_rot, pl.tensor.add(pl.tensor.col_expand_mul(q_hi, cos_hi), pl.tensor.col_expand_mul(q_lo, sin_hi)), [0, 64 // 2])
                            q_nope_latent: pl.Tensor[[1, 512], pl.BFLOAT16] = pl.tensor.matmul(pl.tensor.cast(q_nope, target_type=pl.BFLOAT16, mode=2), pl.tensor.view(w_q_nope_to_latent, [128, 512], [h, 0, 0]), a_trans=False, b_trans=False, c_matrix_nz=False)
                            oi: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.create([1, 512], dtype=pl.FP32)
                            li: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32)
                            mi: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32)
                            oi: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.mul(oi, 0.0)
                            li: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(li, 0.0)
                            mi: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(mi, 0.0)
                            sparse_k: pl.Scalar[pl.INDEX] = min(2048, ctx_len)
                            for kk in pl.range(0, sparse_k, 1):
                                s: pl.Scalar[pl.INDEX] = pl.tensor.read(topk_idx, [0, kk])
                                if s >= 0:
                                    cache_s: pl.Scalar[pl.INDEX] = b * 4096 + s
                                    kv_s: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.cast(pl.tensor.view(kv_cache, [1, 512], [cache_s, 0]), target_type=pl.FP32, mode=2)
                                    pe_s: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(pl.tensor.view(pe_cache, [1, 64], [cache_s, 0]), target_type=pl.FP32, mode=2)
                                    score_nope: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(q_nope_latent, kv_s))
                                    score_pe: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(q_rot, pe_s))
                                    score: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(pl.tensor.add(score_nope, score_pe), 0.0721688)
                                    cur_mi: pl.Tensor[[1, 1], pl.FP32] = score
                                    cur_li: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.exp(pl.tensor.sub(score, cur_mi))
                                    oi_tmp: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.row_expand_mul(kv_s, cur_li)
                                    if kk == 0:
                                        oi: pl.Tensor[[1, 512], pl.FP32] = oi_tmp
                                        li: pl.Tensor[[1, 1], pl.FP32] = cur_li
                                        mi: pl.Tensor[[1, 1], pl.FP32] = cur_mi
                                    else:
                                        mi_new: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.maximum(mi, cur_mi)
                                        alpha: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.exp(pl.tensor.sub(mi, mi_new))
                                        beta: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.exp(pl.tensor.sub(cur_mi, mi_new))
                                        li: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.add(pl.tensor.mul(alpha, li), pl.tensor.mul(beta, cur_li))
                                        oi: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.add(pl.tensor.row_expand_mul(oi, alpha), pl.tensor.row_expand_mul(oi_tmp, beta))
                                        mi: pl.Tensor[[1, 1], pl.FP32] = mi_new
                            ctx_latent: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.row_expand_div(oi, li)
                            v_col: pl.Scalar[pl.INDEX] = h * 128
                            ctx_v: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.create([1, 128], dtype=pl.FP32)
                            ctx_v: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.mul(ctx_v, 0.0)
                            for vb in pl.range(0, 2, 1):
                                v0: pl.Scalar[pl.INDEX] = vb * 64
                                wv_tile: pl.Tensor[[512, 64], pl.BFLOAT16] = pl.tensor.view(w_latent_to_v, [512, 64], [h, 0, v0])
                                v_part: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.matmul(pl.tensor.cast(ctx_latent, target_type=pl.BFLOAT16, mode=2), wv_tile, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                                ctx_v: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(ctx_v, v_part, [0, v0])
                            attn_row: pl.Tensor[[1, 16384], pl.FP32] = pl.tensor.assemble(attn_row, ctx_v, [0, v_col])
                        attn_tile: pl.Tensor[[4, 16384], pl.FP32] = pl.tensor.assemble(attn_tile, attn_row, [ti, 0])
                    for ti in pl.range(0, valid_tok, 1):
                        pos: pl.Scalar[pl.INDEX] = p0 + ti
                        target_node: pl.Scalar[pl.INDEX] = (b + pos + pl.cast(layer_id, pl.INDEX)) % 128
                        token_row: pl.Tensor[[1, 16384], pl.BFLOAT16] = pl.tensor.cast(pl.tensor.view(attn_tile, [1, 16384], [ti, 0]), target_type=pl.BFLOAT16, mode=2)
                        dispatch_buf: pl.Tensor[[128, 16, 4096, 16384], pl.BFLOAT16] = pl.tensor.assemble(dispatch_buf, token_row, [target_node, b, pos, 0])
        return dispatch_buf