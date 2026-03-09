# pypto.program: DeepSeekV32PrefillFront
import pypto.language as pl

@pl.program
class DeepSeekV32PrefillFront:
    @pl.function
    def deepseek_v3_2_prefill_front_layer(self, hidden_states_0: pl.Tensor[[16, 4096, 7168], pl.BFLOAT16], seq_lens_0: pl.Tensor[[16], pl.INT32], layer_id_t_0: pl.Tensor[[1], pl.INT32], rope_cos_0: pl.Tensor[[4096, 64], pl.FP32], rope_sin_0: pl.Tensor[[4096, 64], pl.FP32], kv_cache_0: pl.Tensor[[65536, 512], pl.BFLOAT16], pe_cache_0: pl.Tensor[[65536, 64], pl.BFLOAT16], input_rms_weight_0: pl.Tensor[[1, 7168], pl.FP32], wq_a_0: pl.Tensor[[7168, 1536], pl.BFLOAT16], q_norm_weight_0: pl.Tensor[[1, 1536], pl.FP32], wq_b_0: pl.Tensor[[1536, 24576], pl.BFLOAT16], wkv_a_0: pl.Tensor[[7168, 576], pl.BFLOAT16], kv_norm_weight_0: pl.Tensor[[1, 512], pl.FP32], w_q_nope_to_latent_0: pl.Tensor[[128, 128, 512], pl.BFLOAT16], w_latent_to_v_0: pl.Tensor[[128, 512, 128], pl.BFLOAT16], dispatch_buf_0: pl.Tensor[[128, 16, 4096, 16384], pl.BFLOAT16]) -> pl.Tensor[[128, 16, 4096, 16384], pl.BFLOAT16]:
        layer_id_0: pl.Scalar[pl.INT32] = pl.tensor.read(layer_id_t_0, [0])
        for b_0, (dispatch_buf_iter_1, kv_cache_iter_1, pe_cache_iter_1) in pl.parallel(0, 16, 1, init_values=(dispatch_buf_0, kv_cache_0, pe_cache_0), chunk=4):
            seq_len_b_0: pl.Scalar[pl.INT32] = pl.tensor.read(seq_lens_0, [b_0])
            tok_blocks_0: pl.Scalar[pl.INDEX] = (pl.cast(seq_len_b_0, pl.INDEX) + 4 - 1) // 4
            for p0_idx_0, (dispatch_buf_iter_3, kv_cache_iter_3, pe_cache_iter_3) in pl.range(0, tok_blocks_0, 1, init_values=(dispatch_buf_iter_1, kv_cache_iter_1, pe_cache_iter_1)):
                p0_0: pl.Scalar[pl.INDEX] = p0_idx_0 * 4
                valid_tok_0: pl.Scalar[pl.INDEX] = min(4, pl.cast(seq_len_b_0, pl.INDEX) - p0_0)
                with pl.auto_incore():
                    sq_sum_0: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.create([4, 1], dtype=pl.FP32)
                    sq_sum_1: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.mul(sq_sum_0, 0.0)
                    usage_pad_0: pl.Tensor[[4, 16384], pl.BFLOAT16] = pl.tensor.create([4, 16384], dtype=pl.BFLOAT16)
                    usage_pad_1: pl.Tensor[[4, 16384], pl.FP32] = pl.tensor.mul(usage_pad_0, 0.0)
                    usage_pad_fp_0: pl.Tensor[[4, 16384], pl.FP32] = pl.tensor.cast(usage_pad_1, target_type=pl.FP32, mode=2)
                    usage_pad_sum_0: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.row_sum(usage_pad_fp_0)
                    for kb_0, (sq_sum_iter_2,) in pl.range(0, 14, 1, init_values=(sq_sum_1,)):
                        k0_0: pl.Scalar[pl.INDEX] = kb_0 * 512
                        _t0: pl.Tensor[[4, 512], pl.BFLOAT16] = pl.tensor.view(hidden_states_0, [4, 512], [b_0, p0_0, k0_0])
                        x_chunk_0: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.cast(_t0, target_type=pl.FP32, mode=2)
                        _t1: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.mul(x_chunk_0, x_chunk_0)
                        _t2: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.row_sum(_t1)
                        sq_sum_4: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.add(sq_sum_iter_2, _t2)
                        sq_sum_3: pl.Tensor[[4, 1], pl.FP32] = pl.yield_(sq_sum_4)
                    _t3: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.mul(sq_sum_3, 0.000139509)
                    _t4: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.add(_t3, 1e-06)
                    inv_rms_0: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.rsqrt(_t4)
                    _t5: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.mul(usage_pad_sum_0, 0.0)
                    inv_rms_1: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.add(inv_rms_0, _t5)
                    q_proj_tile_0: pl.Tensor[[4, 128 * 192], pl.BFLOAT16] = pl.tensor.create([4, 128 * 192], dtype=pl.BFLOAT16)
                    kv_a_tile_0: pl.Tensor[[4, 576], pl.BFLOAT16] = pl.tensor.create([4, 576], dtype=pl.BFLOAT16)
                    for ob_0_out, (k0_iter_1_outer, kb_iter_1_outer, q_proj_tile_iter_1_outer, x_chunk_iter_1_outer) in pl.range(0, 12, 1, init_values=(k0_0, kb_0, q_proj_tile_0, x_chunk_0)):
                        for ob_0_in, (k0_iter_1_inner, kb_iter_1_inner, q_proj_tile_iter_1_inner, x_chunk_iter_1_inner) in pl.parallel(0, 8, 1, init_values=(k0_iter_1_outer, kb_iter_1_outer, q_proj_tile_iter_1_outer, x_chunk_iter_1_outer)):
                            q0_0: pl.Scalar[pl.INDEX] = (0 + (ob_0_out * 8 + ob_0_in) * 1) * 256
                            q_acc_0: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.create([4, 256], dtype=pl.FP32)
                            q_acc_1: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.mul(q_acc_0, 0.0)
                            for kb_3, (k0_iter_3, q_acc_iter_2, x_chunk_iter_3) in pl.range(0, 14, 1, init_values=(k0_iter_1_inner, q_acc_1, x_chunk_iter_1_inner)):
                                k0_5: pl.Scalar[pl.INDEX] = kb_3 * 512
                                _t6: pl.Tensor[[4, 512], pl.BFLOAT16] = pl.tensor.view(hidden_states_0, [4, 512], [b_0, p0_0, k0_5])
                                x_chunk_5: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.cast(_t6, target_type=pl.FP32, mode=2)
                                gamma_in_0: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.view(input_rms_weight_0, [1, 512], [0, k0_5])
                                _t7: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.row_expand_mul(x_chunk_5, inv_rms_1)
                                normed_0: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.col_expand_mul(_t7, gamma_in_0)
                                for rb_0, (q_acc_iter_4,) in pl.range(0, 12, 1, init_values=(q_acc_iter_2,)):
                                    r0_0: pl.Scalar[pl.INDEX] = rb_0 * 128
                                    wq_a_chunk_0: pl.Tensor[[512, 128], pl.BFLOAT16] = pl.tensor.view(wq_a_0, [512, 128], [k0_5, r0_0])
                                    _t8: pl.Tensor[[4, 512], pl.BFLOAT16] = pl.tensor.cast(normed_0, target_type=pl.BFLOAT16, mode=2)
                                    qr_part_0: pl.Tensor[[4, 128], pl.BFLOAT16] = pl.tensor.matmul(_t8, wq_a_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                                    gamma_q_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.view(q_norm_weight_0, [1, 128], [0, r0_0])
                                    qn_part_0: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.col_expand_mul(qr_part_0, gamma_q_0)
                                    wq_b_chunk_0: pl.Tensor[[128, 256], pl.BFLOAT16] = pl.tensor.view(wq_b_0, [128, 256], [r0_0, q0_0])
                                    _t9: pl.Tensor[[4, 128], pl.BFLOAT16] = pl.tensor.cast(qn_part_0, target_type=pl.BFLOAT16, mode=2)
                                    _t10: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.matmul(_t9, wq_b_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                                    q_acc_6: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.add(q_acc_iter_4, _t10)
                                    q_acc_5: pl.Tensor[[4, 256], pl.FP32] = pl.yield_(q_acc_6)
                                k0_4, q_acc_3, x_chunk_4 = pl.yield_(k0_5, q_acc_5, x_chunk_5)
                            _t11: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.cast(q_acc_3, target_type=pl.BFLOAT16, mode=2)
                            q_proj_tile_3: pl.Tensor[[4, 128 * 192], pl.BFLOAT16] = pl.tensor.assemble(q_proj_tile_iter_1_inner, _t11, [0, q0_0])
                            k0_iter_1_inner_rv, kb_iter_1_inner_rv, q_proj_tile_iter_1_inner_rv, x_chunk_iter_1_inner_rv = pl.yield_(k0_4, kb_3, q_proj_tile_3, x_chunk_4)
                        k0_iter_1_outer_rv, kb_iter_1_outer_rv, q_proj_tile_iter_1_outer_rv, x_chunk_iter_1_outer_rv = pl.yield_(k0_iter_1_inner_rv, kb_iter_1_inner_rv, q_proj_tile_iter_1_inner_rv, x_chunk_iter_1_inner_rv)
                    for ob_1_rem, (k0_iter_6_rem, kb_iter_4_rem, kv_a_tile_iter_1_rem, normed_iter_1_rem, x_chunk_iter_6_rem) in pl.parallel(0, 5, 1, init_values=(k0_2, kb_2, kv_a_tile_0, normed_0, x_chunk_2)):
                        kv0_0: pl.Scalar[pl.INDEX] = (0 + ob_1_rem * 1) * 128
                        kv_acc_0: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.create([4, 128], dtype=pl.FP32)
                        kv_acc_1: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.mul(kv_acc_0, 0.0)
                        for kb_6, (k0_iter_8, kv_acc_iter_2, normed_iter_3, x_chunk_iter_8) in pl.range(0, 14, 1, init_values=(k0_iter_6_rem, kv_acc_1, normed_iter_1_rem, x_chunk_iter_6_rem)):
                            k0_10: pl.Scalar[pl.INDEX] = kb_6 * 512
                            _t12: pl.Tensor[[4, 512], pl.BFLOAT16] = pl.tensor.view(hidden_states_0, [4, 512], [b_0, p0_0, k0_10])
                            x_chunk_10: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.cast(_t12, target_type=pl.FP32, mode=2)
                            gamma_0: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.view(input_rms_weight_0, [1, 512], [0, k0_10])
                            _t13: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.row_expand_mul(x_chunk_10, inv_rms_1)
                            normed_5: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.col_expand_mul(_t13, gamma_0)
                            wkv_chunk_0: pl.Tensor[[512, 128], pl.BFLOAT16] = pl.tensor.view(wkv_a_0, [512, 128], [k0_10, kv0_0])
                            _t14: pl.Tensor[[4, 512], pl.BFLOAT16] = pl.tensor.cast(normed_5, target_type=pl.BFLOAT16, mode=2)
                            _t15: pl.Tensor[[4, 128], pl.BFLOAT16] = pl.tensor.matmul(_t14, wkv_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                            kv_acc_4: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.add(kv_acc_iter_2, _t15)
                            k0_9, kv_acc_3, normed_4, x_chunk_9 = pl.yield_(k0_10, kv_acc_4, normed_5, x_chunk_10)
                        _t16: pl.Tensor[[4, 128], pl.BFLOAT16] = pl.tensor.cast(kv_acc_3, target_type=pl.BFLOAT16, mode=2)
                        kv_a_tile_3: pl.Tensor[[4, 576], pl.BFLOAT16] = pl.tensor.assemble(kv_a_tile_iter_1_rem, _t16, [0, kv0_0])
                        k0_iter_6_rem_rv, kb_iter_4_rem_rv, kv_a_tile_iter_1_rem_rv, normed_iter_1_rem_rv, x_chunk_iter_6_rem_rv = pl.yield_(k0_9, kb_6, kv_a_tile_3, normed_4, x_chunk_9)
                with pl.auto_incore():
                    attn_tile_0: pl.Tensor[[4, 16384], pl.FP32] = pl.tensor.create([4, 16384], dtype=pl.FP32)
                    attn_tile_1: pl.Tensor[[4, 16384], pl.FP32] = pl.tensor.mul(attn_tile_0, 0.0)
                    for ti_0, (attn_tile_iter_2, kv_cache_iter_5, pe_cache_iter_5) in pl.range(0, valid_tok_0, 1, init_values=(attn_tile_1, kv_cache_iter_3, pe_cache_iter_3)):
                        pos_0: pl.Scalar[pl.INDEX] = p0_0 + ti_0
                        ctx_len_0: pl.Scalar[pl.INDEX] = pos_0 + 1
                        cos_row_0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.view(rope_cos_0, [1, 64], [pos_0, 0])
                        sin_row_0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.view(rope_sin_0, [1, 64], [pos_0, 0])
                        cos_lo_0: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.view(cos_row_0, [1, 64 // 2], [0, 0])
                        cos_hi_0: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.view(cos_row_0, [1, 64 // 2], [0, 64 // 2])
                        sin_lo_0: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.view(sin_row_0, [1, 64 // 2], [0, 0])
                        sin_hi_0: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.view(sin_row_0, [1, 64 // 2], [0, 64 // 2])
                        cache_row_0: pl.Scalar[pl.INDEX] = b_0 * 4096 + pos_0
                        _t17: pl.Tensor[[1, 512], pl.BFLOAT16] = pl.tensor.view(kv_a_tile_iter_1_rem_rv, [1, 512], [ti_0, 0])
                        kv_row_0: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.cast(_t17, target_type=pl.FP32, mode=2)
                        kv_gamma_0: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.view(kv_norm_weight_0, [1, 512], [0, 0])
                        kv_normed_0: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.col_expand_mul(kv_row_0, kv_gamma_0)
                        _t18: pl.Tensor[[1, 64], pl.BFLOAT16] = pl.tensor.view(kv_a_tile_iter_1_rem_rv, [1, 64], [ti_0, 512])
                        pe_row_0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(_t18, target_type=pl.FP32, mode=2)
                        pe_lo_0: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.view(pe_row_0, [1, 64 // 2], [0, 0])
                        pe_hi_0: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.view(pe_row_0, [1, 64 // 2], [0, 64 // 2])
                        pe_rot_0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.create([1, 64], dtype=pl.FP32)
                        _t19: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.col_expand_mul(pe_lo_0, cos_lo_0)
                        _t20: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.col_expand_mul(pe_hi_0, sin_lo_0)
                        _t21: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.sub(_t19, _t20)
                        pe_rot_1: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.assemble(pe_rot_0, _t21, [0, 0])
                        _t22: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.col_expand_mul(pe_hi_0, cos_hi_0)
                        _t23: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.col_expand_mul(pe_lo_0, sin_hi_0)
                        _t24: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.add(_t22, _t23)
                        pe_rot_2: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.assemble(pe_rot_1, _t24, [0, 64 // 2])
                        _t25: pl.Tensor[[1, 512], pl.BFLOAT16] = pl.tensor.cast(kv_normed_0, target_type=pl.BFLOAT16, mode=2)
                        kv_cache_7: pl.Tensor[[65536, 512], pl.BFLOAT16] = pl.tensor.assemble(kv_cache_iter_5, _t25, [cache_row_0, 0])
                        _t26: pl.Tensor[[1, 64], pl.BFLOAT16] = pl.tensor.cast(pe_rot_2, target_type=pl.BFLOAT16, mode=2)
                        pe_cache_7: pl.Tensor[[65536, 64], pl.BFLOAT16] = pl.tensor.assemble(pe_cache_iter_5, _t26, [cache_row_0, 0])
                        topk_vals_0: pl.Tensor[[1, 2048], pl.FP32] = pl.tensor.create([1, 2048], dtype=pl.FP32)
                        topk_idx_0: pl.Tensor[[1, 2048], pl.INT32] = pl.tensor.create([1, 2048], dtype=pl.INT32)
                        blk_topk_vals_0: pl.Tensor[[2, 2048], pl.FP32] = pl.tensor.create([2, 2048], dtype=pl.FP32)
                        blk_topk_idx_0: pl.Tensor[[2, 2048], pl.INT32] = pl.tensor.create([2, 2048], dtype=pl.INT32)
                        topk_vals_1: pl.Tensor[[1, 2048], pl.FP32] = pl.tensor.mul(topk_vals_0, -340282299999999994960115009090224128000.0)
                        topk_idx_1: pl.Tensor[[1, 2048], pl.INDEX] = pl.tensor.mul(topk_idx_0, 0)
                        blk_topk_vals_1: pl.Tensor[[2, 2048], pl.FP32] = pl.tensor.mul(blk_topk_vals_0, -340282299999999994960115009090224128000.0)
                        blk_topk_idx_1: pl.Tensor[[2, 2048], pl.INDEX] = pl.tensor.mul(blk_topk_idx_0, 0)
                        for kk_0, (blk_topk_idx_iter_2, topk_idx_iter_2) in pl.range(0, 2048, 1, init_values=(blk_topk_idx_1, topk_idx_1)):
                            neg_one_0: pl.Tensor[[1, 1], pl.INT32] = pl.tensor.create([1, 1], dtype=pl.INT32)
                            neg_one_1: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.mul(neg_one_0, 0)
                            neg_one_2: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.add(neg_one_1, -1)
                            topk_idx_4: pl.Tensor[[1, 2048], pl.INDEX] = pl.tensor.assemble(topk_idx_iter_2, neg_one_2, [0, kk_0])
                            blk_topk_idx_4: pl.Tensor[[2, 2048], pl.INDEX] = pl.tensor.assemble(blk_topk_idx_iter_2, neg_one_2, [0, kk_0])
                            blk_topk_idx_5: pl.Tensor[[2, 2048], pl.INDEX] = pl.tensor.assemble(blk_topk_idx_4, neg_one_2, [1, kk_0])
                            blk_topk_idx_3, topk_idx_3 = pl.yield_(blk_topk_idx_5, topk_idx_4)
                        q_col0_0: pl.Scalar[pl.INDEX] = 0
                        _t27: pl.Tensor[[1, 128], pl.BFLOAT16] = pl.tensor.view(q_proj_tile_iter_1_outer_rv, [1, 128], [ti_0, q_col0_0])
                        q_nope0_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.cast(_t27, target_type=pl.FP32, mode=2)
                        _t28: pl.Tensor[[1, 64], pl.BFLOAT16] = pl.tensor.view(q_proj_tile_iter_1_outer_rv, [1, 64], [ti_0, q_col0_0 + 128])
                        q_pe0_0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(_t28, target_type=pl.FP32, mode=2)
                        q0_lo_0: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.view(q_pe0_0, [1, 64 // 2], [0, 0])
                        q0_hi_0: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.view(q_pe0_0, [1, 64 // 2], [0, 64 // 2])
                        q0_rot_0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.create([1, 64], dtype=pl.FP32)
                        _t29: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.col_expand_mul(q0_lo_0, cos_lo_0)
                        _t30: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.col_expand_mul(q0_hi_0, sin_lo_0)
                        _t31: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.sub(_t29, _t30)
                        q0_rot_1: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.assemble(q0_rot_0, _t31, [0, 0])
                        _t32: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.col_expand_mul(q0_hi_0, cos_hi_0)
                        _t33: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.col_expand_mul(q0_lo_0, sin_hi_0)
                        _t34: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.add(_t32, _t33)
                        q0_rot_2: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.assemble(q0_rot_1, _t34, [0, 64 // 2])
                        _t35: pl.Tensor[[1, 128], pl.BFLOAT16] = pl.tensor.cast(q_nope0_0, target_type=pl.BFLOAT16, mode=2)
                        _t36: pl.Tensor[[128, 512], pl.BFLOAT16] = pl.tensor.view(w_q_nope_to_latent_0, [128, 512], [0, 0, 0])
                        q0_nope_latent_0: pl.Tensor[[1, 512], pl.BFLOAT16] = pl.tensor.matmul(_t35, _t36, a_trans=False, b_trans=False, c_matrix_nz=False)
                        sparse_k_gen_0: pl.Scalar[pl.INDEX] = min(2048, ctx_len_0)
                        for blk_0, (blk_topk_idx_iter_6, blk_topk_vals_iter_2, kk_iter_1) in pl.range(0, 2, 1, init_values=(blk_topk_idx_3, blk_topk_vals_1, kk_0)):
                            blk_start_0: pl.Scalar[pl.INDEX] = blk_0 * 2048
                            blk_end_0: pl.Scalar[pl.INDEX] = min(ctx_len_0, blk_start_0 + 2048)
                            for ss_0, (blk_topk_idx_iter_8, blk_topk_vals_iter_4, kk_iter_3) in pl.range(0, 2048, 1, init_values=(blk_topk_idx_iter_6, blk_topk_vals_iter_2, kk_iter_1)):
                                s_0: pl.Scalar[pl.INDEX] = blk_start_0 + ss_0
                                if s_0 < blk_end_0:
                                    cache_s_0: pl.Scalar[pl.INDEX] = b_0 * 4096 + s_0
                                    _t37: pl.Tensor[[1, 512], pl.BFLOAT16] = pl.tensor.view(kv_cache_7, [1, 512], [cache_s_0, 0])
                                    kv_s_0: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.cast(_t37, target_type=pl.FP32, mode=2)
                                    _t38: pl.Tensor[[1, 64], pl.BFLOAT16] = pl.tensor.view(pe_cache_7, [1, 64], [cache_s_0, 0])
                                    pe_s_0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(_t38, target_type=pl.FP32, mode=2)
                                    _t39: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.mul(q0_nope_latent_0, kv_s_0)
                                    score_nope_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.row_sum(_t39)
                                    _t40: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.mul(q0_rot_2, pe_s_0)
                                    score_pe_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.row_sum(_t40)
                                    _t41: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.add(score_nope_0, score_pe_0)
                                    score_fp32_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(_t41, 0.0721688)
                                    score_fp8_0: pl.Tensor[[1, 1], pl.FP8E4M3FN] = pl.tensor.cast(score_fp32_0, target_type=pl.FP8E4M3FN, mode=2)
                                    score_a5_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.cast(score_fp8_0, target_type=pl.FP32, mode=2)
                                    cur_score_0: pl.Scalar[pl.FP32] = pl.tensor.read(score_a5_0, [0, 0])
                                    inserted_0: pl.Tensor[[1, 1], pl.INT32] = pl.tensor.create([1, 1], dtype=pl.INT32)
                                    inserted_1: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.mul(inserted_0, 0)
                                    for kk_5, (blk_topk_idx_iter_10, blk_topk_vals_iter_6, inserted_iter_2) in pl.range(0, sparse_k_gen_0, 1, init_values=(blk_topk_idx_iter_8, blk_topk_vals_iter_4, inserted_1)):
                                        ins_0: pl.Scalar[pl.INDEX] = pl.tensor.read(inserted_iter_2, [0, 0])
                                        kth_val_0: pl.Scalar[pl.FP32] = pl.tensor.read(blk_topk_vals_iter_6, [blk_0, kk_5])
                                        if ins_0 == 0:
                                            if cur_score_0 > kth_val_0:
                                                for sh_0, (blk_topk_idx_iter_12, blk_topk_vals_iter_8) in pl.range(sparse_k_gen_0 - 1, kk_5, -1, init_values=(blk_topk_idx_iter_10, blk_topk_vals_iter_6)):
                                                    prev_val_0: pl.Scalar[pl.FP32] = pl.tensor.read(blk_topk_vals_iter_8, [blk_0, sh_0 - 1])
                                                    prev_idx_0: pl.Scalar[pl.INDEX] = pl.tensor.read(blk_topk_idx_iter_12, [blk_0, sh_0 - 1])
                                                    prev_val_t_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32)
                                                    prev_idx_t_0: pl.Tensor[[1, 1], pl.INT32] = pl.tensor.create([1, 1], dtype=pl.INT32)
                                                    prev_val_t_1: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(prev_val_t_0, 0.0)
                                                    prev_idx_t_1: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.mul(prev_idx_t_0, 0)
                                                    prev_val_t_2: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.add(prev_val_t_1, prev_val_0)
                                                    prev_idx_t_2: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.add(prev_idx_t_1, prev_idx_0)
                                                    blk_topk_vals_10: pl.Tensor[[2, 2048], pl.FP32] = pl.tensor.assemble(blk_topk_vals_iter_8, prev_val_t_2, [blk_0, sh_0])
                                                    blk_topk_idx_14: pl.Tensor[[2, 2048], pl.INDEX] = pl.tensor.assemble(blk_topk_idx_iter_12, prev_idx_t_2, [blk_0, sh_0])
                                                    blk_topk_idx_13, blk_topk_vals_9 = pl.yield_(blk_topk_idx_14, blk_topk_vals_10)
                                                cur_score_t_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32)
                                                cur_index_t_0: pl.Tensor[[1, 1], pl.INT32] = pl.tensor.create([1, 1], dtype=pl.INT32)
                                                one_t_0: pl.Tensor[[1, 1], pl.INT32] = pl.tensor.create([1, 1], dtype=pl.INT32)
                                                cur_score_t_1: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(cur_score_t_0, 0.0)
                                                cur_index_t_1: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.mul(cur_index_t_0, 0)
                                                one_t_1: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.mul(one_t_0, 0)
                                                cur_score_t_2: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.add(cur_score_t_1, cur_score_0)
                                                cur_index_t_2: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.add(cur_index_t_1, s_0)
                                                one_t_2: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.add(one_t_1, 1)
                                                blk_topk_vals_11: pl.Tensor[[2, 2048], pl.FP32] = pl.tensor.assemble(blk_topk_vals_9, cur_score_t_2, [blk_0, kk_5])
                                                blk_topk_idx_15: pl.Tensor[[2, 2048], pl.INDEX] = pl.tensor.assemble(blk_topk_idx_13, cur_index_t_2, [blk_0, kk_5])
                                                inserted_4: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.assemble(inserted_iter_2, one_t_2, [0, 0])
                                                blk_topk_idx_16, blk_topk_vals_12, inserted_5 = pl.yield_(blk_topk_idx_15, blk_topk_vals_11, inserted_4)
                                            else:
                                                blk_topk_idx_16, blk_topk_vals_12, inserted_5 = pl.yield_(blk_topk_idx_iter_10, blk_topk_vals_iter_6, inserted_iter_2)
                                            blk_topk_idx_17, blk_topk_vals_13, inserted_6 = pl.yield_(blk_topk_idx_16, blk_topk_vals_12, inserted_5)
                                        else:
                                            blk_topk_idx_17, blk_topk_vals_13, inserted_6 = pl.yield_(blk_topk_idx_iter_10, blk_topk_vals_iter_6, inserted_iter_2)
                                        blk_topk_idx_11, blk_topk_vals_7, inserted_3 = pl.yield_(blk_topk_idx_17, blk_topk_vals_13, inserted_6)
                                    blk_topk_idx_18, blk_topk_vals_14, kk_6 = pl.yield_(blk_topk_idx_11, blk_topk_vals_7, kk_5)
                                else:
                                    blk_topk_idx_18, blk_topk_vals_14, kk_6 = pl.yield_(blk_topk_idx_iter_8, blk_topk_vals_iter_4, kk_iter_3)
                                blk_topk_idx_9, blk_topk_vals_5, kk_4 = pl.yield_(blk_topk_idx_18, blk_topk_vals_14, kk_6)
                            blk_topk_idx_7, blk_topk_vals_3, kk_2 = pl.yield_(blk_topk_idx_9, blk_topk_vals_5, kk_4)
                        for blk_1, (kk_iter_7, topk_idx_iter_5, topk_vals_iter_2) in pl.range(0, 2, 1, init_values=(kk_2, topk_idx_3, topk_vals_1)):
                            for kk_9, (topk_idx_iter_7, topk_vals_iter_4) in pl.range(0, sparse_k_gen_0, 1, init_values=(topk_idx_iter_5, topk_vals_iter_2)):
                                cand_idx_0: pl.Scalar[pl.INDEX] = pl.tensor.read(blk_topk_idx_7, [blk_1, kk_9])
                                if cand_idx_0 >= 0:
                                    cand_val_0: pl.Scalar[pl.FP32] = pl.tensor.read(blk_topk_vals_3, [blk_1, kk_9])
                                    inserted_7: pl.Tensor[[1, 1], pl.INT32] = pl.tensor.create([1, 1], dtype=pl.INT32)
                                    inserted_8: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.mul(inserted_7, 0)
                                    for tkk_0, (inserted_iter_9, topk_idx_iter_9, topk_vals_iter_6) in pl.range(0, sparse_k_gen_0, 1, init_values=(inserted_8, topk_idx_iter_7, topk_vals_iter_4)):
                                        ins_1: pl.Scalar[pl.INDEX] = pl.tensor.read(inserted_iter_9, [0, 0])
                                        kth_val_1: pl.Scalar[pl.FP32] = pl.tensor.read(topk_vals_iter_6, [0, tkk_0])
                                        if ins_1 == 0:
                                            if cand_val_0 > kth_val_1:
                                                for sh_1, (topk_idx_iter_11, topk_vals_iter_8) in pl.range(sparse_k_gen_0 - 1, tkk_0, -1, init_values=(topk_idx_iter_9, topk_vals_iter_6)):
                                                    prev_val_1: pl.Scalar[pl.FP32] = pl.tensor.read(topk_vals_iter_8, [0, sh_1 - 1])
                                                    prev_idx_1: pl.Scalar[pl.INDEX] = pl.tensor.read(topk_idx_iter_11, [0, sh_1 - 1])
                                                    prev_val_t_3: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32)
                                                    prev_idx_t_3: pl.Tensor[[1, 1], pl.INT32] = pl.tensor.create([1, 1], dtype=pl.INT32)
                                                    prev_val_t_4: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(prev_val_t_3, 0.0)
                                                    prev_idx_t_4: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.mul(prev_idx_t_3, 0)
                                                    prev_val_t_5: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.add(prev_val_t_4, prev_val_1)
                                                    prev_idx_t_5: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.add(prev_idx_t_4, prev_idx_1)
                                                    topk_vals_10: pl.Tensor[[1, 2048], pl.FP32] = pl.tensor.assemble(topk_vals_iter_8, prev_val_t_5, [0, sh_1])
                                                    topk_idx_13: pl.Tensor[[1, 2048], pl.INDEX] = pl.tensor.assemble(topk_idx_iter_11, prev_idx_t_5, [0, sh_1])
                                                    topk_idx_12, topk_vals_9 = pl.yield_(topk_idx_13, topk_vals_10)
                                                cand_val_t_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32)
                                                cand_idx_t_0: pl.Tensor[[1, 1], pl.INT32] = pl.tensor.create([1, 1], dtype=pl.INT32)
                                                one_t_3: pl.Tensor[[1, 1], pl.INT32] = pl.tensor.create([1, 1], dtype=pl.INT32)
                                                cand_val_t_1: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(cand_val_t_0, 0.0)
                                                cand_idx_t_1: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.mul(cand_idx_t_0, 0)
                                                one_t_4: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.mul(one_t_3, 0)
                                                cand_val_t_2: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.add(cand_val_t_1, cand_val_0)
                                                cand_idx_t_2: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.add(cand_idx_t_1, cand_idx_0)
                                                one_t_5: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.add(one_t_4, 1)
                                                topk_vals_11: pl.Tensor[[1, 2048], pl.FP32] = pl.tensor.assemble(topk_vals_9, cand_val_t_2, [0, tkk_0])
                                                topk_idx_14: pl.Tensor[[1, 2048], pl.INDEX] = pl.tensor.assemble(topk_idx_12, cand_idx_t_2, [0, tkk_0])
                                                inserted_11: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.assemble(inserted_iter_9, one_t_5, [0, 0])
                                                inserted_12, topk_idx_15, topk_vals_12 = pl.yield_(inserted_11, topk_idx_14, topk_vals_11)
                                            else:
                                                inserted_12, topk_idx_15, topk_vals_12 = pl.yield_(inserted_iter_9, topk_idx_iter_9, topk_vals_iter_6)
                                            inserted_13, topk_idx_16, topk_vals_13 = pl.yield_(inserted_12, topk_idx_15, topk_vals_12)
                                        else:
                                            inserted_13, topk_idx_16, topk_vals_13 = pl.yield_(inserted_iter_9, topk_idx_iter_9, topk_vals_iter_6)
                                        inserted_10, topk_idx_10, topk_vals_7 = pl.yield_(inserted_13, topk_idx_16, topk_vals_13)
                                    topk_idx_17, topk_vals_14 = pl.yield_(topk_idx_10, topk_vals_7)
                                else:
                                    topk_idx_17, topk_vals_14 = pl.yield_(topk_idx_iter_7, topk_vals_iter_4)
                                topk_idx_8, topk_vals_5 = pl.yield_(topk_idx_17, topk_vals_14)
                            kk_8, topk_idx_6, topk_vals_3 = pl.yield_(kk_9, topk_idx_8, topk_vals_5)
                        attn_row_0: pl.Tensor[[1, 16384], pl.FP32] = pl.tensor.create([1, 16384], dtype=pl.FP32)
                        attn_row_1: pl.Tensor[[1, 16384], pl.FP32] = pl.tensor.mul(attn_row_0, 0.0)
                        for h_0_out, (attn_row_iter_2_outer, kk_iter_10_outer, s_iter_1_outer) in pl.range(0, 16, 1, init_values=(attn_row_1, kk_8, s_0)):
                            for h_0_in, (attn_row_iter_2_inner, kk_iter_10_inner, s_iter_1_inner) in pl.parallel(0, 8, 1, init_values=(attn_row_iter_2_outer, kk_iter_10_outer, s_iter_1_outer)):
                                q_col_0: pl.Scalar[pl.INDEX] = (0 + (h_0_out * 8 + h_0_in) * 1) * 192
                                _t42: pl.Tensor[[1, 128], pl.BFLOAT16] = pl.tensor.view(q_proj_tile_iter_1_outer_rv, [1, 128], [ti_0, q_col_0])
                                q_nope_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.cast(_t42, target_type=pl.FP32, mode=2)
                                _t43: pl.Tensor[[1, 64], pl.BFLOAT16] = pl.tensor.view(q_proj_tile_iter_1_outer_rv, [1, 64], [ti_0, q_col_0 + 128])
                                q_pe_0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(_t43, target_type=pl.FP32, mode=2)
                                q_lo_0: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.view(q_pe_0, [1, 64 // 2], [0, 0])
                                q_hi_0: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.view(q_pe_0, [1, 64 // 2], [0, 64 // 2])
                                q_rot_0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.create([1, 64], dtype=pl.FP32)
                                _t44: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.col_expand_mul(q_lo_0, cos_lo_0)
                                _t45: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.col_expand_mul(q_hi_0, sin_lo_0)
                                _t46: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.sub(_t44, _t45)
                                q_rot_1: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.assemble(q_rot_0, _t46, [0, 0])
                                _t47: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.col_expand_mul(q_hi_0, cos_hi_0)
                                _t48: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.col_expand_mul(q_lo_0, sin_hi_0)
                                _t49: pl.Tensor[[1, 64 // 2], pl.FP32] = pl.tensor.add(_t47, _t48)
                                q_rot_2: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.assemble(q_rot_1, _t49, [0, 64 // 2])
                                _t50: pl.Tensor[[1, 128], pl.BFLOAT16] = pl.tensor.cast(q_nope_0, target_type=pl.BFLOAT16, mode=2)
                                _t51: pl.Tensor[[128, 512], pl.BFLOAT16] = pl.tensor.view(w_q_nope_to_latent_0, [128, 512], [0 + (h_0_out * 8 + h_0_in) * 1, 0, 0])
                                q_nope_latent_0: pl.Tensor[[1, 512], pl.BFLOAT16] = pl.tensor.matmul(_t50, _t51, a_trans=False, b_trans=False, c_matrix_nz=False)
                                oi_0: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.create([1, 512], dtype=pl.FP32)
                                li_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32)
                                mi_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32)
                                oi_1: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.mul(oi_0, 0.0)
                                li_1: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(li_0, 0.0)
                                mi_1: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(mi_0, 0.0)
                                sparse_k_0: pl.Scalar[pl.INDEX] = min(2048, ctx_len_0)
                                for kk_12, (li_iter_2, mi_iter_2, oi_iter_2, s_iter_3) in pl.range(0, sparse_k_0, 1, init_values=(li_1, mi_1, oi_1, s_iter_1_inner)):
                                    s_5: pl.Scalar[pl.INDEX] = pl.tensor.read(topk_idx_6, [0, kk_12])
                                    if s_5 >= 0:
                                        cache_s_1: pl.Scalar[pl.INDEX] = b_0 * 4096 + s_5
                                        _t52: pl.Tensor[[1, 512], pl.BFLOAT16] = pl.tensor.view(kv_cache_7, [1, 512], [cache_s_1, 0])
                                        kv_s_1: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.cast(_t52, target_type=pl.FP32, mode=2)
                                        _t53: pl.Tensor[[1, 64], pl.BFLOAT16] = pl.tensor.view(pe_cache_7, [1, 64], [cache_s_1, 0])
                                        pe_s_1: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(_t53, target_type=pl.FP32, mode=2)
                                        _t54: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.mul(q_nope_latent_0, kv_s_1)
                                        score_nope_1: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.row_sum(_t54)
                                        _t55: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.mul(q_rot_2, pe_s_1)
                                        score_pe_1: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.row_sum(_t55)
                                        _t56: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.add(score_nope_1, score_pe_1)
                                        score_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(_t56, 0.0721688)
                                        cur_mi_0: pl.Tensor[[1, 1], pl.FP32] = score_0
                                        _t57: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.sub(score_0, cur_mi_0)
                                        cur_li_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.exp(_t57)
                                        oi_tmp_0: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.row_expand_mul(kv_s_1, cur_li_0)
                                        if kk_12 == 0:
                                            oi_4: pl.Tensor[[1, 512], pl.FP32] = oi_tmp_0
                                            li_4: pl.Tensor[[1, 1], pl.FP32] = cur_li_0
                                            mi_4: pl.Tensor[[1, 1], pl.FP32] = cur_mi_0
                                            li_6, mi_6, oi_6 = pl.yield_(li_4, mi_4, oi_4)
                                        else:
                                            mi_new_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.maximum(mi_iter_2, cur_mi_0)
                                            _t58: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.sub(mi_iter_2, mi_new_0)
                                            alpha_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.exp(_t58)
                                            _t59: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.sub(cur_mi_0, mi_new_0)
                                            beta_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.exp(_t59)
                                            _t60: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(alpha_0, li_iter_2)
                                            _t61: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(beta_0, cur_li_0)
                                            li_5: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.add(_t60, _t61)
                                            _t62: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.row_expand_mul(oi_iter_2, alpha_0)
                                            _t63: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.row_expand_mul(oi_tmp_0, beta_0)
                                            oi_5: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.add(_t62, _t63)
                                            mi_5: pl.Tensor[[1, 1], pl.FP32] = mi_new_0
                                            li_6, mi_6, oi_6 = pl.yield_(li_5, mi_5, oi_5)
                                        li_7, mi_7, oi_7 = pl.yield_(li_6, mi_6, oi_6)
                                    else:
                                        li_7, mi_7, oi_7 = pl.yield_(li_iter_2, mi_iter_2, oi_iter_2)
                                    li_3, mi_3, oi_3, s_4 = pl.yield_(li_7, mi_7, oi_7, s_5)
                                ctx_latent_0: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.row_expand_div(oi_3, li_3)
                                v_col_0: pl.Scalar[pl.INDEX] = (0 + (h_0_out * 8 + h_0_in) * 1) * 128
                                ctx_v_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.create([1, 128], dtype=pl.FP32)
                                ctx_v_1: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.mul(ctx_v_0, 0.0)
                                for vb_0, (ctx_v_iter_2,) in pl.range(0, 2, 1, init_values=(ctx_v_1,)):
                                    v0_0: pl.Scalar[pl.INDEX] = vb_0 * 64
                                    wv_tile_0: pl.Tensor[[512, 64], pl.BFLOAT16] = pl.tensor.view(w_latent_to_v_0, [512, 64], [0 + (h_0_out * 8 + h_0_in) * 1, 0, v0_0])
                                    _t64: pl.Tensor[[1, 512], pl.BFLOAT16] = pl.tensor.cast(ctx_latent_0, target_type=pl.BFLOAT16, mode=2)
                                    v_part_0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.matmul(_t64, wv_tile_0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                                    ctx_v_4: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(ctx_v_iter_2, v_part_0, [0, v0_0])
                                    ctx_v_3: pl.Tensor[[1, 128], pl.FP32] = pl.yield_(ctx_v_4)
                                attn_row_4: pl.Tensor[[1, 16384], pl.FP32] = pl.tensor.assemble(attn_row_iter_2_inner, ctx_v_3, [0, v_col_0])
                                attn_row_iter_2_inner_rv, kk_iter_10_inner_rv, s_iter_1_inner_rv = pl.yield_(attn_row_4, kk_12, s_4)
                            attn_row_iter_2_outer_rv, kk_iter_10_outer_rv, s_iter_1_outer_rv = pl.yield_(attn_row_iter_2_inner_rv, kk_iter_10_inner_rv, s_iter_1_inner_rv)
                        attn_tile_4: pl.Tensor[[4, 16384], pl.FP32] = pl.tensor.assemble(attn_tile_iter_2, attn_row_iter_2_outer_rv, [ti_0, 0])
                        attn_tile_3, kv_cache_6, pe_cache_6 = pl.yield_(attn_tile_4, kv_cache_7, pe_cache_7)
                    for ti_1, (dispatch_buf_iter_5, pos_iter_1) in pl.range(0, valid_tok_0, 1, init_values=(dispatch_buf_iter_3, pos_0)):
                        pos_3: pl.Scalar[pl.INDEX] = p0_0 + ti_1
                        target_node_0: pl.Scalar[pl.INDEX] = (b_0 + pos_3 + pl.cast(layer_id_0, pl.INDEX)) % 128
                        _t65: pl.Tensor[[1, 16384], pl.FP32] = pl.tensor.view(attn_tile_3, [1, 16384], [ti_1, 0])
                        token_row_0: pl.Tensor[[1, 16384], pl.BFLOAT16] = pl.tensor.cast(_t65, target_type=pl.BFLOAT16, mode=2)
                        dispatch_buf_7: pl.Tensor[[128, 16, 4096, 16384], pl.BFLOAT16] = pl.tensor.assemble(dispatch_buf_iter_5, token_row_0, [target_node_0, b_0, pos_3, 0])
                        dispatch_buf_6, pos_2 = pl.yield_(dispatch_buf_7, pos_3)
                dispatch_buf_4, kv_cache_4, pe_cache_4 = pl.yield_(dispatch_buf_6, kv_cache_6, pe_cache_6)
            dispatch_buf_2, kv_cache_2, pe_cache_2 = pl.yield_(dispatch_buf_4, kv_cache_4, pe_cache_4)
        return dispatch_buf_2