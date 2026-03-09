# pypto.program: DeepSeekV32DecodeFront
import pypto.language as pl

@pl.program
class DeepSeekV32DecodeFront:
    @pl.function(type=pl.FunctionType.Orchestration)
    def deepseek_v3_2_decode_front_layer(self, hidden_states_0: pl.Tensor[[16, 7168], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 229376, 0)], seq_lens_0: pl.Tensor[[16], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 1)], layer_id_t_0: pl.Tensor[[1], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 2)], rope_cos_0: pl.Tensor[[4096, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 1048576, 3)], rope_sin_0: pl.Tensor[[4096, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 1048576, 4)], kv_cache_0: pl.Tensor[[65536, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 67108864, 5)], pe_cache_0: pl.Tensor[[65536, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 8388608, 6)], input_rms_weight_0: pl.Tensor[[1, 7168], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 28672, 7)], wq_a_0: pl.Tensor[[7168, 1536], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 22020096, 8)], q_norm_weight_0: pl.Tensor[[1, 1536], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 6144, 9)], wq_b_0: pl.Tensor[[1536, 24576], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 75497472, 10)], wkv_a_0: pl.Tensor[[7168, 576], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 8257536, 11)], kv_norm_weight_0: pl.Tensor[[1, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 12)], w_q_nope_to_latent_0: pl.Tensor[[128, 128, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16777216, 13)], w_latent_to_v_0: pl.Tensor[[128, 512, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16777216, 14)], dispatch_buf_0: pl.Tensor[[128, 16, 16384], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 67108864, 15)]) -> pl.Tensor[[128, 16, 16384], pl.BFLOAT16]:
        layer_id_0: pl.Scalar[pl.INT32] = pl.tensor.read(layer_id_t_0, [0])
        qr_0: pl.Tensor[[16, 1536], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 49152, 16)] = pl.tensor.create([16, 1536], dtype=pl.BFLOAT16)
        q_proj_0: pl.Tensor[[16, 128 * 192], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 17)] = pl.tensor.create([16, 128 * 192], dtype=pl.BFLOAT16)
        kv_a_0: pl.Tensor[[16, 576], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 18432, 18)] = pl.tensor.create([16, 576], dtype=pl.BFLOAT16)
        attn_front_0: pl.Tensor[[16, 16384], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 1048576, 19)] = pl.tensor.create([16, 16384], dtype=pl.FP32)
        sq_sum_0: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 20)] = pl.tensor.create([16, 1], dtype=pl.FP32)
        sq_sum_1: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 21)] = pl.tensor.mul(sq_sum_0, 0.0)
        usage_pad_0: pl.Tensor[[4, 16384], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 131072, 22)] = pl.tensor.create([4, 16384], dtype=pl.BFLOAT16)
        usage_pad_1: pl.Tensor[[4, 16384], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 262144, 23)] = pl.tensor.mul(usage_pad_0, 0.0)
        usage_pad_fp_0: pl.Tensor[[4, 16384], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 262144, 24)] = pl.tensor.cast(usage_pad_1, target_type=pl.FP32, mode=2)
        usage_pad_sum_0: pl.Tensor[[4, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 16, 25)] = pl.tensor.row_sum(usage_pad_fp_0)
        for kb_0, (sq_sum_iter_2,) in pl.range(0, 14, 1, init_values=(sq_sum_1,)):
            k0_0: pl.Scalar[pl.INDEX] = kb_0 * 512
            _t0: pl.Tensor[[16, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 26)] = pl.tensor.view(hidden_states_0, [16, 512], [0, k0_0])
            x_chunk_0: pl.Tensor[[16, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 27)] = pl.tensor.cast(_t0, target_type=pl.FP32, mode=2)
            _t1: pl.Tensor[[16, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 28)] = pl.tensor.mul(x_chunk_0, x_chunk_0)
            _t2: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 29)] = pl.tensor.row_sum(_t1)
            sq_sum_4: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 30)] = pl.tensor.add(sq_sum_iter_2, _t2)
            sq_sum_3: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 31)] = pl.yield_(sq_sum_4)
        _t3: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 32)] = pl.tensor.mul(sq_sum_3, 0.000139509)
        _t4: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 33)] = pl.tensor.add(_t3, 1e-06)
        inv_rms_0: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 34)] = pl.tensor.rsqrt(_t4)
        for b0_0, (k0_iter_1, kb_iter_1, kv_a_iter_1, q_proj_iter_1, qr_iter_1, x_chunk_iter_1) in pl.range(0, 16, 4, init_values=(k0_0, kb_0, kv_a_0, q_proj_0, qr_0, x_chunk_0)):
            inv_rms_tile_0: pl.Tensor[[4, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 16, 35)] = pl.tensor.view(inv_rms_0, [4, 1], [b0_0, 0])
            _t5: pl.Tensor[[4, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 16, 36)] = pl.tensor.mul(usage_pad_sum_0, 0.0)
            inv_rms_tile_1: pl.Tensor[[4, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 16, 37)] = pl.tensor.add(inv_rms_tile_0, _t5)
            for ob_0_out, (k0_iter_3_outer_l0, kb_iter_3_outer_l0, qr_iter_3_outer_l0, x_chunk_iter_3_outer_l0) in pl.range(0, 3, 1, init_values=(k0_iter_1, kb_iter_1, qr_iter_1, x_chunk_iter_1)):
                ret: pl.Tuple([pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Tensor[[16, 1536], pl.BFLOAT16], pl.Tensor[[16, 512], pl.FP32]]) = self.call_group(deepseek_v3_2_decode_front_layer_incore_0_group, b0_0, hidden_states_0, input_rms_weight_0, inv_rms_tile_1, k0_0, k0_iter_1, k0_iter_3_outer_l0, kb_0, kb_iter_1, kb_iter_3_outer_l0, ob_0_out, qr_0, qr_iter_1, qr_iter_3_outer_l0, wq_a_0, x_chunk_0, x_chunk_iter_1, x_chunk_iter_3_outer_l0)
                k0_iter_3_outer_l1_rv: pl.Scalar[pl.INDEX] = ret[0]
                kb_iter_3_outer_l1_rv: pl.Scalar[pl.INDEX] = ret[1]
                qr_iter_3_outer_l1_rv: pl.Tensor[[16, 1536], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 49152, 38)] = ret[2]
                x_chunk_iter_3_outer_l1_rv: pl.Tensor[[16, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 39)] = ret[3]
                k0_iter_3_outer_l0_rv, kb_iter_3_outer_l0_rv, qr_iter_3_outer_l0_rv, x_chunk_iter_3_outer_l0_rv = pl.yield_(k0_iter_3_outer_l1_rv, kb_iter_3_outer_l1_rv, qr_iter_3_outer_l1_rv, x_chunk_iter_3_outer_l1_rv)
            for ob_1_out, (gamma_iter_1_outer_l0, k0_iter_8_outer_l0, kb_iter_6_outer_l0, q0_iter_1_outer_l0, q_acc_iter_5_outer_l0, q_proj_iter_3_outer_l0, wq_chunk_iter_1_outer_l0) in pl.range(0, 6, 1, init_values=(gamma_0, k0_4, kb_4, q0_0, q_acc_3, q_proj_iter_1, wq_chunk_0)):
                ret: pl.Tuple([pl.Tensor[[1, 512], pl.FP32], pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Tensor[[4, 128], pl.FP32], pl.Tensor[[16, 128 * 192], pl.BFLOAT16], pl.Tensor[[512, 128], pl.BFLOAT16]]) = self.call_group(deepseek_v3_2_decode_front_layer_incore_1_group, b0_0, gamma_0, gamma_iter_1_outer_l0, k0_4, k0_iter_8_outer_l0, kb_4, kb_iter_6_outer_l0, ob_1_out, q0_0, q0_iter_1_outer_l0, q_acc_3, q_acc_iter_5_outer_l0, q_norm_weight_0, q_proj_0, q_proj_iter_1, q_proj_iter_3_outer_l0, qr_iter_3_outer_l0_rv, wq_b_0, wq_chunk_0, wq_chunk_iter_1_outer_l0)
                gamma_iter_1_outer_l1_rv: pl.Tensor[[1, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 45)] = ret[0]
                k0_iter_8_outer_l1_rv: pl.Scalar[pl.INDEX] = ret[1]
                kb_iter_6_outer_l1_rv: pl.Scalar[pl.INDEX] = ret[2]
                q0_iter_1_outer_l1_rv: pl.Scalar[pl.INDEX] = ret[3]
                q_acc_iter_5_outer_l1_rv: pl.Tensor[[4, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 46)] = ret[4]
                q_proj_iter_3_outer_l1_rv: pl.Tensor[[16, 128 * 192], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 47)] = ret[5]
                wq_chunk_iter_1_outer_l1_rv: pl.Tensor[[512, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 131072, 48)] = ret[6]
                gamma_iter_1_outer_l0_rv, k0_iter_8_outer_l0_rv, kb_iter_6_outer_l0_rv, q0_iter_1_outer_l0_rv, q_acc_iter_5_outer_l0_rv, q_proj_iter_3_outer_l0_rv, wq_chunk_iter_1_outer_l0_rv = pl.yield_(gamma_iter_1_outer_l1_rv, k0_iter_8_outer_l1_rv, kb_iter_6_outer_l1_rv, q0_iter_1_outer_l1_rv, q_acc_iter_5_outer_l1_rv, q_proj_iter_3_outer_l1_rv, wq_chunk_iter_1_outer_l1_rv)
            for ob_2_rem, (gamma_iter_6_rem, k0_iter_13_rem, kb_iter_9_rem, kv_a_iter_3_rem, normed_iter_1_rem, x_chunk_iter_8_rem, x_chunk_bf16_iter_1_rem) in pl.parallel(0, 5, 1, init_values=(gamma_2, k0_9, kb_7, kv_a_iter_1, normed_0, x_chunk_4, x_chunk_bf16_0)):
                kv0_0: pl.Scalar[pl.INDEX] = (0 + ob_2_rem * 1) * 128
                kv_acc_0: pl.Tensor[[4, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 57)] = pl.tensor.create([4, 128], dtype=pl.FP32)
                kv_acc_1: pl.Tensor[[4, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 58)] = pl.tensor.mul(kv_acc_0, 0.0)
                for kb_11, (gamma_iter_8, k0_iter_15, kv_acc_iter_2, normed_iter_3, x_chunk_iter_10, x_chunk_bf16_iter_3) in pl.range(0, 14, 1, init_values=(gamma_iter_6_rem, k0_iter_13_rem, kv_acc_1, normed_iter_1_rem, x_chunk_iter_8_rem, x_chunk_bf16_iter_1_rem)):
                    k0_17: pl.Scalar[pl.INDEX] = kb_11 * 512
                    x_chunk_bf16_5: pl.Tensor[[4, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 59)] = pl.tensor.view(hidden_states_0, [4, 512], [b0_0, k0_17])
                    x_chunk_12: pl.Tensor[[4, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 60)] = pl.tensor.cast(x_chunk_bf16_5, target_type=pl.FP32, mode=2)
                    gamma_10: pl.Tensor[[1, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 61)] = pl.tensor.view(input_rms_weight_0, [1, 512], [0, k0_17])
                    _t14: pl.Tensor[[4, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 62)] = pl.tensor.row_expand_mul(x_chunk_12, inv_rms_tile_1)
                    normed_5: pl.Tensor[[4, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 63)] = pl.tensor.col_expand_mul(_t14, gamma_10)
                    wkv_chunk_0: pl.Tensor[[512, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 131072, 64)] = pl.tensor.view(wkv_a_0, [512, 128], [k0_17, kv0_0])
                    _t15: pl.Tensor[[4, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 65)] = pl.tensor.cast(normed_5, target_type=pl.BFLOAT16, mode=2)
                    _t16: pl.Tensor[[4, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 66)] = pl.tensor.matmul(_t15, wkv_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                    kv_acc_4: pl.Tensor[[4, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 67)] = pl.tensor.add(kv_acc_iter_2, _t16)
                    gamma_9, k0_16, kv_acc_3, normed_4, x_chunk_11, x_chunk_bf16_4 = pl.yield_(gamma_10, k0_17, kv_acc_4, normed_5, x_chunk_12, x_chunk_bf16_5)
                _t17: pl.Tensor[[4, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 73)] = pl.tensor.cast(kv_acc_3, target_type=pl.BFLOAT16, mode=2)
                kv_a_5: pl.Tensor[[16, 576], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 18432, 74)] = pl.tensor.assemble(kv_a_iter_3_rem, _t17, [b0_0, kv0_0])
                gamma_iter_6_rem_rv, k0_iter_13_rem_rv, kb_iter_9_rem_rv, kv_a_iter_3_rem_rv, normed_iter_1_rem_rv, x_chunk_iter_8_rem_rv, x_chunk_bf16_iter_1_rem_rv = pl.yield_(gamma_9, k0_16, kb_11, kv_a_5, normed_4, x_chunk_11, x_chunk_bf16_4)
            k0_2, kb_2, kv_a_2, q_proj_2, qr_2, x_chunk_2 = pl.yield_(k0_iter_13_rem_rv, kb_iter_9_rem_rv, kv_a_iter_3_rem_rv, q_proj_iter_3_outer_l0_rv, qr_iter_3_outer_l0_rv, x_chunk_iter_8_rem_rv)
        for b_0, (attn_front_iter_1, kv_cache_iter_1, pe_cache_iter_1) in pl.parallel(0, 16, 1, init_values=(attn_front_0, kv_cache_0, pe_cache_0), chunk=4):
            ctx_len_0: pl.Scalar[pl.INT32] = pl.tensor.read(seq_lens_0, [b_0])
            pos_0: pl.Scalar[pl.INDEX] = pl.cast(ctx_len_0, pl.INDEX) - 1
            cos_row_0: pl.Tensor[[1, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 84)] = pl.tensor.view(rope_cos_0, [1, 64], [pos_0, 0])
            sin_row_0: pl.Tensor[[1, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 85)] = pl.tensor.view(rope_sin_0, [1, 64], [pos_0, 0])
            cos_lo_0: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 86)] = pl.tensor.view(cos_row_0, [1, 64 // 2], [0, 0])
            cos_hi_0: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 87)] = pl.tensor.view(cos_row_0, [1, 64 // 2], [0, 64 // 2])
            sin_lo_0: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 88)] = pl.tensor.view(sin_row_0, [1, 64 // 2], [0, 0])
            sin_hi_0: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 89)] = pl.tensor.view(sin_row_0, [1, 64 // 2], [0, 64 // 2])
            cache_row_0: pl.Scalar[pl.INDEX] = b_0 * 4096 + pos_0
            _t18: pl.Tensor[[1, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 90)] = pl.tensor.view(kv_a_2, [1, 512], [b_0, 0])
            kv_row_0: pl.Tensor[[1, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 91)] = pl.tensor.cast(_t18, target_type=pl.FP32, mode=2)
            kv_gamma_0: pl.Tensor[[1, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 92)] = pl.tensor.view(kv_norm_weight_0, [1, 512], [0, 0])
            kv_normed_0: pl.Tensor[[1, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 93)] = pl.tensor.col_expand_mul(kv_row_0, kv_gamma_0)
            _t19: pl.Tensor[[1, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 128, 94)] = pl.tensor.view(kv_a_2, [1, 64], [b_0, 512])
            pe_row_0: pl.Tensor[[1, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 95)] = pl.tensor.cast(_t19, target_type=pl.FP32, mode=2)
            pe_lo_0: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 96)] = pl.tensor.view(pe_row_0, [1, 64 // 2], [0, 0])
            pe_hi_0: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 97)] = pl.tensor.view(pe_row_0, [1, 64 // 2], [0, 64 // 2])
            pe_rot_0: pl.Tensor[[1, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 98)] = pl.tensor.create([1, 64], dtype=pl.FP32)
            _t20: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 99)] = pl.tensor.col_expand_mul(pe_lo_0, cos_lo_0)
            _t21: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 100)] = pl.tensor.col_expand_mul(pe_hi_0, sin_lo_0)
            _t22: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 101)] = pl.tensor.sub(_t20, _t21)
            pe_rot_1: pl.Tensor[[1, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 102)] = pl.tensor.assemble(pe_rot_0, _t22, [0, 0])
            _t23: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 103)] = pl.tensor.col_expand_mul(pe_hi_0, cos_hi_0)
            _t24: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 104)] = pl.tensor.col_expand_mul(pe_lo_0, sin_hi_0)
            _t25: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 105)] = pl.tensor.add(_t23, _t24)
            pe_rot_2: pl.Tensor[[1, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 106)] = pl.tensor.assemble(pe_rot_1, _t25, [0, 64 // 2])
            _t26: pl.Tensor[[1, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 107)] = pl.tensor.cast(kv_normed_0, target_type=pl.BFLOAT16, mode=2)
            kv_cache_3: pl.Tensor[[65536, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 67108864, 108)] = pl.tensor.assemble(kv_cache_iter_1, _t26, [cache_row_0, 0])
            _t27: pl.Tensor[[1, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 128, 109)] = pl.tensor.cast(pe_rot_2, target_type=pl.BFLOAT16, mode=2)
            pe_cache_3: pl.Tensor[[65536, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 8388608, 110)] = pl.tensor.assemble(pe_cache_iter_1, _t27, [cache_row_0, 0])
            topk_vals_0: pl.Tensor[[1, 2048], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 111)] = pl.tensor.create([1, 2048], dtype=pl.FP32)
            topk_idx_0: pl.Tensor[[1, 2048], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 112)] = pl.tensor.create([1, 2048], dtype=pl.INT32)
            blk_topk_vals_0: pl.Tensor[[2, 2048], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 113)] = pl.tensor.create([2, 2048], dtype=pl.FP32)
            blk_topk_idx_0: pl.Tensor[[2, 2048], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 114)] = pl.tensor.create([2, 2048], dtype=pl.INT32)
            topk_vals_1: pl.Tensor[[1, 2048], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 115)] = pl.tensor.mul(topk_vals_0, -340282299999999994960115009090224128000.0)
            topk_idx_1: pl.Tensor[[1, 2048], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 116)] = pl.tensor.mul(topk_idx_0, 0)
            blk_topk_vals_1: pl.Tensor[[2, 2048], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 117)] = pl.tensor.mul(blk_topk_vals_0, -340282299999999994960115009090224128000.0)
            blk_topk_idx_1: pl.Tensor[[2, 2048], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 118)] = pl.tensor.mul(blk_topk_idx_0, 0)
            for kk_0, (blk_topk_idx_iter_2, topk_idx_iter_2) in pl.range(0, 2048, 1, init_values=(blk_topk_idx_1, topk_idx_1)):
                neg_one_0: pl.Tensor[[1, 1], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 119)] = pl.tensor.create([1, 1], dtype=pl.INT32)
                neg_one_1: pl.Tensor[[1, 1], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 8, 120)] = pl.tensor.mul(neg_one_0, 0)
                neg_one_2: pl.Tensor[[1, 1], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 8, 121)] = pl.tensor.add(neg_one_1, -1)
                topk_idx_4: pl.Tensor[[1, 2048], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 122)] = pl.tensor.assemble(topk_idx_iter_2, neg_one_2, [0, kk_0])
                blk_topk_idx_4: pl.Tensor[[2, 2048], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 123)] = pl.tensor.assemble(blk_topk_idx_iter_2, neg_one_2, [0, kk_0])
                blk_topk_idx_5: pl.Tensor[[2, 2048], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 124)] = pl.tensor.assemble(blk_topk_idx_4, neg_one_2, [1, kk_0])
                blk_topk_idx_3, topk_idx_3 = pl.yield_(blk_topk_idx_5, topk_idx_4)
            q_col0_0: pl.Scalar[pl.INDEX] = 0
            _t28: pl.Tensor[[1, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 127)] = pl.tensor.view(q_proj_2, [1, 128], [b_0, q_col0_0])
            q_nope0_0: pl.Tensor[[1, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 128)] = pl.tensor.cast(_t28, target_type=pl.FP32, mode=2)
            _t29: pl.Tensor[[1, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 128, 129)] = pl.tensor.view(q_proj_2, [1, 64], [b_0, q_col0_0 + 128])
            q_pe0_0: pl.Tensor[[1, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 130)] = pl.tensor.cast(_t29, target_type=pl.FP32, mode=2)
            q0_lo_0: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 131)] = pl.tensor.view(q_pe0_0, [1, 64 // 2], [0, 0])
            q0_hi_0: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 132)] = pl.tensor.view(q_pe0_0, [1, 64 // 2], [0, 64 // 2])
            q0_rot_0: pl.Tensor[[1, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 133)] = pl.tensor.create([1, 64], dtype=pl.FP32)
            _t30: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 134)] = pl.tensor.col_expand_mul(q0_lo_0, cos_lo_0)
            _t31: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 135)] = pl.tensor.col_expand_mul(q0_hi_0, sin_lo_0)
            _t32: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 136)] = pl.tensor.sub(_t30, _t31)
            q0_rot_1: pl.Tensor[[1, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 137)] = pl.tensor.assemble(q0_rot_0, _t32, [0, 0])
            _t33: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 138)] = pl.tensor.col_expand_mul(q0_hi_0, cos_hi_0)
            _t34: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 139)] = pl.tensor.col_expand_mul(q0_lo_0, sin_hi_0)
            _t35: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 140)] = pl.tensor.add(_t33, _t34)
            q0_rot_2: pl.Tensor[[1, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 141)] = pl.tensor.assemble(q0_rot_1, _t35, [0, 64 // 2])
            _t36: pl.Tensor[[1, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 142)] = pl.tensor.cast(q_nope0_0, target_type=pl.BFLOAT16, mode=2)
            _t37: pl.Tensor[[128, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 131072, 143)] = pl.tensor.view(w_q_nope_to_latent_0, [128, 512], [0, 0, 0])
            q0_nope_latent_0: pl.Tensor[[1, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 144)] = pl.tensor.matmul(_t36, _t37, a_trans=False, b_trans=False, c_matrix_nz=False)
            sparse_k_gen_0: pl.Scalar[pl.INDEX] = min(2048, pl.cast(ctx_len_0, pl.INDEX))
            for blk_0, (blk_topk_idx_iter_6, blk_topk_vals_iter_2, kk_iter_1) in pl.range(0, 2, 1, init_values=(blk_topk_idx_3, blk_topk_vals_1, kk_0)):
                blk_start_0: pl.Scalar[pl.INDEX] = blk_0 * 2048
                blk_end_0: pl.Scalar[pl.INDEX] = min(pl.cast(ctx_len_0, pl.INDEX), blk_start_0 + 2048)
                for ss_0, (blk_topk_idx_iter_8, blk_topk_vals_iter_4, kk_iter_3) in pl.range(0, 2048, 1, init_values=(blk_topk_idx_iter_6, blk_topk_vals_iter_2, kk_iter_1)):
                    s_0: pl.Scalar[pl.INDEX] = blk_start_0 + ss_0
                    if s_0 < blk_end_0:
                        cache_s_0: pl.Scalar[pl.INDEX] = b_0 * 4096 + s_0
                        _t38: pl.Tensor[[1, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 145)] = pl.tensor.view(kv_cache_3, [1, 512], [cache_s_0, 0])
                        kv_s_0: pl.Tensor[[1, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 146)] = pl.tensor.cast(_t38, target_type=pl.FP32, mode=2)
                        _t39: pl.Tensor[[1, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 128, 147)] = pl.tensor.view(pe_cache_3, [1, 64], [cache_s_0, 0])
                        pe_s_0: pl.Tensor[[1, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 148)] = pl.tensor.cast(_t39, target_type=pl.FP32, mode=2)
                        _t40: pl.Tensor[[1, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 149)] = pl.tensor.mul(q0_nope_latent_0, kv_s_0)
                        score_nope_0: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 150)] = pl.tensor.row_sum(_t40)
                        _t41: pl.Tensor[[1, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 151)] = pl.tensor.mul(q0_rot_2, pe_s_0)
                        score_pe_0: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 152)] = pl.tensor.row_sum(_t41)
                        _t42: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 153)] = pl.tensor.add(score_nope_0, score_pe_0)
                        score_fp32_0: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 154)] = pl.tensor.mul(_t42, 0.0721688)
                        score_fp8_0: pl.Tensor[[1, 1], pl.FP8E4M3FN, pl.MemRef(pl.MemorySpace.DDR, -1, 1, 155)] = pl.tensor.cast(score_fp32_0, target_type=pl.FP8E4M3FN, mode=2)
                        score_a5_0: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 156)] = pl.tensor.cast(score_fp8_0, target_type=pl.FP32, mode=2)
                        cur_score_0: pl.Scalar[pl.FP32] = pl.tensor.read(score_a5_0, [0, 0])
                        inserted_0: pl.Tensor[[1, 1], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 157)] = pl.tensor.create([1, 1], dtype=pl.INT32)
                        inserted_1: pl.Tensor[[1, 1], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 8, 158)] = pl.tensor.mul(inserted_0, 0)
                        for kk_5, (blk_topk_idx_iter_10, blk_topk_vals_iter_6, inserted_iter_2) in pl.range(0, sparse_k_gen_0, 1, init_values=(blk_topk_idx_iter_8, blk_topk_vals_iter_4, inserted_1)):
                            ins_0: pl.Scalar[pl.INDEX] = pl.tensor.read(inserted_iter_2, [0, 0])
                            kth_val_0: pl.Scalar[pl.FP32] = pl.tensor.read(blk_topk_vals_iter_6, [blk_0, kk_5])
                            if ins_0 == 0:
                                if cur_score_0 > kth_val_0:
                                    for sh_0, (blk_topk_idx_iter_12, blk_topk_vals_iter_8) in pl.range(sparse_k_gen_0 - 1, kk_5, -1, init_values=(blk_topk_idx_iter_10, blk_topk_vals_iter_6)):
                                        prev_val_0: pl.Scalar[pl.FP32] = pl.tensor.read(blk_topk_vals_iter_8, [blk_0, sh_0 - 1])
                                        prev_idx_0: pl.Scalar[pl.INDEX] = pl.tensor.read(blk_topk_idx_iter_12, [blk_0, sh_0 - 1])
                                        prev_val_t_0: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 159)] = pl.tensor.create([1, 1], dtype=pl.FP32)
                                        prev_idx_t_0: pl.Tensor[[1, 1], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 160)] = pl.tensor.create([1, 1], dtype=pl.INT32)
                                        prev_val_t_1: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 161)] = pl.tensor.mul(prev_val_t_0, 0.0)
                                        prev_idx_t_1: pl.Tensor[[1, 1], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 8, 162)] = pl.tensor.mul(prev_idx_t_0, 0)
                                        prev_val_t_2: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 163)] = pl.tensor.add(prev_val_t_1, prev_val_0)
                                        prev_idx_t_2: pl.Tensor[[1, 1], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 8, 164)] = pl.tensor.add(prev_idx_t_1, prev_idx_0)
                                        blk_topk_vals_10: pl.Tensor[[2, 2048], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 165)] = pl.tensor.assemble(blk_topk_vals_iter_8, prev_val_t_2, [blk_0, sh_0])
                                        blk_topk_idx_14: pl.Tensor[[2, 2048], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 166)] = pl.tensor.assemble(blk_topk_idx_iter_12, prev_idx_t_2, [blk_0, sh_0])
                                        blk_topk_idx_13, blk_topk_vals_9 = pl.yield_(blk_topk_idx_14, blk_topk_vals_10)
                                    cur_score_t_0: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 169)] = pl.tensor.create([1, 1], dtype=pl.FP32)
                                    cur_index_t_0: pl.Tensor[[1, 1], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 170)] = pl.tensor.create([1, 1], dtype=pl.INT32)
                                    one_t_0: pl.Tensor[[1, 1], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 171)] = pl.tensor.create([1, 1], dtype=pl.INT32)
                                    cur_score_t_1: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 172)] = pl.tensor.mul(cur_score_t_0, 0.0)
                                    cur_index_t_1: pl.Tensor[[1, 1], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 8, 173)] = pl.tensor.mul(cur_index_t_0, 0)
                                    one_t_1: pl.Tensor[[1, 1], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 8, 174)] = pl.tensor.mul(one_t_0, 0)
                                    cur_score_t_2: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 175)] = pl.tensor.add(cur_score_t_1, cur_score_0)
                                    cur_index_t_2: pl.Tensor[[1, 1], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 8, 176)] = pl.tensor.add(cur_index_t_1, s_0)
                                    one_t_2: pl.Tensor[[1, 1], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 8, 177)] = pl.tensor.add(one_t_1, 1)
                                    blk_topk_vals_11: pl.Tensor[[2, 2048], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 178)] = pl.tensor.assemble(blk_topk_vals_9, cur_score_t_2, [blk_0, kk_5])
                                    blk_topk_idx_15: pl.Tensor[[2, 2048], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 179)] = pl.tensor.assemble(blk_topk_idx_13, cur_index_t_2, [blk_0, kk_5])
                                    inserted_4: pl.Tensor[[1, 1], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 8, 180)] = pl.tensor.assemble(inserted_iter_2, one_t_2, [0, 0])
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
                        inserted_7: pl.Tensor[[1, 1], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 196)] = pl.tensor.create([1, 1], dtype=pl.INT32)
                        inserted_8: pl.Tensor[[1, 1], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 8, 197)] = pl.tensor.mul(inserted_7, 0)
                        for tkk_0, (inserted_iter_9, topk_idx_iter_9, topk_vals_iter_6) in pl.range(0, sparse_k_gen_0, 1, init_values=(inserted_8, topk_idx_iter_7, topk_vals_iter_4)):
                            ins_1: pl.Scalar[pl.INDEX] = pl.tensor.read(inserted_iter_9, [0, 0])
                            kth_val_1: pl.Scalar[pl.FP32] = pl.tensor.read(topk_vals_iter_6, [0, tkk_0])
                            if ins_1 == 0:
                                if cand_val_0 > kth_val_1:
                                    for sh_1, (topk_idx_iter_11, topk_vals_iter_8) in pl.range(sparse_k_gen_0 - 1, tkk_0, -1, init_values=(topk_idx_iter_9, topk_vals_iter_6)):
                                        prev_val_1: pl.Scalar[pl.FP32] = pl.tensor.read(topk_vals_iter_8, [0, sh_1 - 1])
                                        prev_idx_1: pl.Scalar[pl.INDEX] = pl.tensor.read(topk_idx_iter_11, [0, sh_1 - 1])
                                        prev_val_t_3: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 198)] = pl.tensor.create([1, 1], dtype=pl.FP32)
                                        prev_idx_t_3: pl.Tensor[[1, 1], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 199)] = pl.tensor.create([1, 1], dtype=pl.INT32)
                                        prev_val_t_4: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 200)] = pl.tensor.mul(prev_val_t_3, 0.0)
                                        prev_idx_t_4: pl.Tensor[[1, 1], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 8, 201)] = pl.tensor.mul(prev_idx_t_3, 0)
                                        prev_val_t_5: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 202)] = pl.tensor.add(prev_val_t_4, prev_val_1)
                                        prev_idx_t_5: pl.Tensor[[1, 1], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 8, 203)] = pl.tensor.add(prev_idx_t_4, prev_idx_1)
                                        topk_vals_10: pl.Tensor[[1, 2048], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 204)] = pl.tensor.assemble(topk_vals_iter_8, prev_val_t_5, [0, sh_1])
                                        topk_idx_13: pl.Tensor[[1, 2048], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 205)] = pl.tensor.assemble(topk_idx_iter_11, prev_idx_t_5, [0, sh_1])
                                        topk_idx_12, topk_vals_9 = pl.yield_(topk_idx_13, topk_vals_10)
                                    cand_val_t_0: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 208)] = pl.tensor.create([1, 1], dtype=pl.FP32)
                                    cand_idx_t_0: pl.Tensor[[1, 1], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 209)] = pl.tensor.create([1, 1], dtype=pl.INT32)
                                    one_t_3: pl.Tensor[[1, 1], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 210)] = pl.tensor.create([1, 1], dtype=pl.INT32)
                                    cand_val_t_1: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 211)] = pl.tensor.mul(cand_val_t_0, 0.0)
                                    cand_idx_t_1: pl.Tensor[[1, 1], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 8, 212)] = pl.tensor.mul(cand_idx_t_0, 0)
                                    one_t_4: pl.Tensor[[1, 1], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 8, 213)] = pl.tensor.mul(one_t_3, 0)
                                    cand_val_t_2: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 214)] = pl.tensor.add(cand_val_t_1, cand_val_0)
                                    cand_idx_t_2: pl.Tensor[[1, 1], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 8, 215)] = pl.tensor.add(cand_idx_t_1, cand_idx_0)
                                    one_t_5: pl.Tensor[[1, 1], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 8, 216)] = pl.tensor.add(one_t_4, 1)
                                    topk_vals_11: pl.Tensor[[1, 2048], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 217)] = pl.tensor.assemble(topk_vals_9, cand_val_t_2, [0, tkk_0])
                                    topk_idx_14: pl.Tensor[[1, 2048], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 218)] = pl.tensor.assemble(topk_idx_12, cand_idx_t_2, [0, tkk_0])
                                    inserted_11: pl.Tensor[[1, 1], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 8, 219)] = pl.tensor.assemble(inserted_iter_9, one_t_5, [0, 0])
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
            attn_row_0: pl.Tensor[[1, 16384], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 235)] = pl.tensor.create([1, 16384], dtype=pl.FP32)
            attn_row_1: pl.Tensor[[1, 16384], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 236)] = pl.tensor.mul(attn_row_0, 0.0)
            for h_0_out, (attn_row_iter_2_outer_l0, kk_iter_10_outer_l0, s_iter_1_outer_l0) in pl.range(0, 16, 1, init_values=(attn_row_1, kk_8, s_0)):
                ret: pl.Tuple([pl.Tensor[[1, 16384], pl.FP32], pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX]]) = self.call_group(deepseek_v3_2_decode_front_layer_incore_2_group, attn_row_1, attn_row_iter_2_outer_l0, b_0, cos_hi_0, cos_lo_0, ctx_len_0, h_0_out, kk_8, kk_iter_10_outer_l0, kv_cache_3, pe_cache_3, q_proj_2, s_0, s_iter_1_outer_l0, sin_hi_0, sin_lo_0, topk_idx_6, w_latent_to_v_0, w_q_nope_to_latent_0)
                attn_row_iter_2_outer_l1_rv: pl.Tensor[[1, 16384], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 237)] = ret[0]
                kk_iter_10_outer_l1_rv: pl.Scalar[pl.INDEX] = ret[1]
                s_iter_1_outer_l1_rv: pl.Scalar[pl.INDEX] = ret[2]
                attn_row_iter_2_outer_l0_rv, kk_iter_10_outer_l0_rv, s_iter_1_outer_l0_rv = pl.yield_(attn_row_iter_2_outer_l1_rv, kk_iter_10_outer_l1_rv, s_iter_1_outer_l1_rv)
            attn_front_3: pl.Tensor[[16, 16384], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 1048576, 239)] = pl.tensor.assemble(attn_front_iter_1, attn_row_iter_2_outer_l0_rv, [b_0, 0])
            attn_front_2, kv_cache_2, pe_cache_2 = pl.yield_(attn_front_3, kv_cache_3, pe_cache_3)
        for b_1, (dispatch_buf_iter_1,) in pl.parallel(0, 16, 1, init_values=(dispatch_buf_0,), chunk=4):
            target_node_0: pl.Scalar[pl.INDEX] = (b_1 + pl.cast(layer_id_0, pl.INDEX)) % 128
            _t66: pl.Tensor[[1, 16384], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 243)] = pl.tensor.view(attn_front_2, [1, 16384], [b_1, 0])
            token_row_0: pl.Tensor[[1, 16384], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 244)] = pl.tensor.cast(_t66, target_type=pl.BFLOAT16, mode=2)
            dispatch_buf_3: pl.Tensor[[128, 16, 16384], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 67108864, 245)] = pl.tensor.assemble(dispatch_buf_iter_1, token_row_0, [target_node_0, b_1, 0])
            dispatch_buf_2: pl.Tensor[[128, 16, 16384], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 67108864, 246)] = pl.yield_(dispatch_buf_3)
        return dispatch_buf_2
    @pl.function(type=pl.FunctionType.InCore)
    def deepseek_v3_2_decode_front_layer_incore_0_aic(self, b0_0: pl.Scalar[pl.INDEX], hidden_states_0: pl.Tensor[[16, 7168], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 229376, 0)], input_rms_weight_0: pl.Tensor[[1, 7168], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 28672, 1)], inv_rms_tile_1: pl.Tensor[[4, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 16, 2)], k0_0: pl.Scalar[pl.INDEX], k0_iter_1: pl.Scalar[pl.INDEX], k0_iter_3_outer_l0: pl.Scalar[pl.INDEX], kb_0: pl.Scalar[pl.INDEX], kb_iter_1: pl.Scalar[pl.INDEX], kb_iter_3_outer_l0: pl.Scalar[pl.INDEX], ob_0_out: pl.Scalar[pl.INDEX], qr_0: pl.Tensor[[16, 1536], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 49152, 3)], qr_iter_1: pl.Tensor[[16, 1536], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 49152, 4)], qr_iter_3_outer_l0: pl.Tensor[[16, 1536], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 49152, 5)], wq_a_0: pl.Tensor[[7168, 1536], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 22020096, 6)], x_chunk_0: pl.Tensor[[16, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 7)], x_chunk_iter_1: pl.Tensor[[16, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 8)], x_chunk_iter_3_outer_l0: pl.Tensor[[16, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 9)]) -> tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Tensor[[16, 1536], pl.BFLOAT16], pl.Tensor[[16, 512], pl.FP32]]:
        pl.comm.aic_initialize_pipe()
        for ob_0_in, (k0_iter_3_outer_l1, kb_iter_3_outer_l1, qr_iter_3_outer_l1, x_chunk_iter_3_outer_l1) in pl.parallel(0, 4, 1, init_values=(k0_iter_3_outer_l0, kb_iter_3_outer_l0, qr_iter_3_outer_l0, x_chunk_iter_3_outer_l0)):
            q0_0: pl.Scalar[pl.INDEX] = (0 + (ob_0_out * 4 + ob_0_in) * 1) * 128
            for kb_5, (k0_iter_5, x_chunk_iter_5) in pl.range(0, 14, 1, init_values=(k0_iter_3_outer_l1, x_chunk_iter_3_outer_l1)):
                k0_7: pl.Scalar[pl.INDEX] = kb_5 * 512
                wq_chunk_0__h0: pl.Tensor[[256, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 10)] = pl.comm.tpop_from_aiv(0)
                wq_chunk_0__h1: pl.Tensor[[256, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 11)] = pl.comm.tpop_from_aiv(1)
                wq_chunk_0__tmp: pl.Tensor[[512, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 131072, 12)] = pl.tensor.create(__list__(512, 128), dtype=pl.BFLOAT16)
                wq_chunk_0__mid: pl.Tensor[[512, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 131072, 13)] = pl.tensor.assemble(wq_chunk_0__tmp, wq_chunk_0__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                wq_chunk_0: pl.Tensor[[512, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 131072, 14)] = pl.tensor.assemble(wq_chunk_0__mid, wq_chunk_0__h1, __list__(256, 0))
                pl.comm.tfree_to_aiv(1)
                _t7__h0: pl.Tensor[[2, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 15)] = pl.comm.tpop_from_aiv(0)
                _t7__h1: pl.Tensor[[2, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 16)] = pl.comm.tpop_from_aiv(1)
                _t7__tmp: pl.Tensor[[4, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 17)] = pl.tensor.create(__list__(4, 512), dtype=pl.BFLOAT16)
                _t7__mid: pl.Tensor[[4, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 18)] = pl.tensor.assemble(_t7__tmp, _t7__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                _t7: pl.Tensor[[4, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 19)] = pl.tensor.assemble(_t7__mid, _t7__h1, __list__(2, 0))
                pl.comm.tfree_to_aiv(1)
                _t8: pl.Tensor[[4, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 20)] = pl.tensor.matmul(_t7, wq_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                __half0__: pl.Tensor[[2, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 21)] = pl.tensor.view(_t8, __list__(2, 128), __list__(0, 0))
                __half1__: pl.Tensor[[2, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 22)] = pl.tensor.view(_t8, __list__(2, 128), __list__(2, 0))
                pl.comm.tpush_to_aiv(__half0__, 0)
                pl.comm.tpush_to_aiv(__half1__, 1)
                k0_6, x_chunk_6 = pl.yield_(k0_7)
            k0_iter_3_outer_l1_rv, kb_iter_3_outer_l1_rv, qr_iter_3_outer_l1_rv, x_chunk_iter_3_outer_l1_rv = pl.yield_(k0_6, kb_5, x_chunk_6)
        return k0_iter_3_outer_l1_rv, kb_iter_3_outer_l1_rv, qr_iter_3_outer_l1_rv, x_chunk_iter_3_outer_l1_rv
    @pl.function(type=pl.FunctionType.InCore)
    def deepseek_v3_2_decode_front_layer_incore_0_aiv(self, b0_0: pl.Scalar[pl.INDEX], hidden_states_0: pl.Tensor[[16, 7168], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 229376, 0)], input_rms_weight_0: pl.Tensor[[1, 7168], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 28672, 1)], inv_rms_tile_1: pl.Tensor[[4, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 16, 2)], k0_0: pl.Scalar[pl.INDEX], k0_iter_1: pl.Scalar[pl.INDEX], k0_iter_3_outer_l0: pl.Scalar[pl.INDEX], kb_0: pl.Scalar[pl.INDEX], kb_iter_1: pl.Scalar[pl.INDEX], kb_iter_3_outer_l0: pl.Scalar[pl.INDEX], ob_0_out: pl.Scalar[pl.INDEX], qr_0: pl.Tensor[[16, 1536], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 49152, 3)], qr_iter_1: pl.Tensor[[16, 1536], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 49152, 4)], qr_iter_3_outer_l0: pl.Tensor[[16, 1536], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 49152, 5)], wq_a_0: pl.Tensor[[7168, 1536], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 22020096, 6)], x_chunk_0: pl.Tensor[[16, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 7)], x_chunk_iter_1: pl.Tensor[[16, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 8)], x_chunk_iter_3_outer_l0: pl.Tensor[[16, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 9)], AIV_IDX: pl.Scalar[pl.INDEX]) -> tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Tensor[[16, 1536], pl.BFLOAT16], pl.Tensor[[16, 512], pl.FP32]]:
        mem_vec_10: pl.MemRefType = pl.block.alloc(pl.MemorySpace.Vec, -1, 16, 10)
        inv_rms_tile_1_tile: pl.Tile[[4, 1], pl.FP32, tile_view=pl.TileView(valid_shape=[4, 1], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null), pl.MemRef(pl.MemorySpace.Vec, -1, 16, 10)] = pl.block.load(inv_rms_tile_1, [0, 0], [4, 1], [4, 1], target_memory=pl.MemorySpace.Vec)
        pl.comm.aiv_initialize_pipe()
        for ob_0_in, (k0_iter_3_outer_l1, kb_iter_3_outer_l1, qr_iter_3_outer_l1, x_chunk_iter_3_outer_l1) in pl.parallel(0, 4, 1, init_values=(k0_iter_3_outer_l0, kb_iter_3_outer_l0, qr_iter_3_outer_l0, x_chunk_iter_3_outer_l0)):
            q0_0: pl.Scalar[pl.INDEX] = (0 + (ob_0_out * 4 + ob_0_in) * 1) * 128
            q_acc_0: pl.Tensor[[2, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 11)] = pl.tensor.create([2, 128], dtype=pl.FP32)
            q_acc_1: pl.Tensor[[4, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 13)] = pl.tensor.mul(q_acc_0, 0.0)
            for kb_5, (k0_iter_5, q_acc_iter_2, x_chunk_iter_5) in pl.range(0, 14, 1, init_values=(k0_iter_3_outer_l1, q_acc_1, x_chunk_iter_3_outer_l1)):
                k0_7: pl.Scalar[pl.INDEX] = kb_5 * 512
                x_chunk_bf16_0: pl.Tensor[[4, 256], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 14)] = pl.tensor.view(hidden_states_0, [4, 256], [b0_0, k0_7 + AIV_IDX * 256])
                x_chunk_7: pl.Tensor[[4, 256], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 16)] = pl.tensor.cast(x_chunk_bf16_0, target_type=pl.FP32, mode=2)
                gamma_0: pl.Tensor[[1, 256], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 17)] = pl.tensor.view(input_rms_weight_0, [1, 256], [0, k0_7 + AIV_IDX * 256])
                _t6: pl.Tensor[[4, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 19)] = pl.tensor.row_expand_mul(x_chunk_7, inv_rms_tile_1)
                normed_0: pl.Tensor[[4, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 21)] = pl.tensor.col_expand_mul(_t6, gamma_0)
                wq_chunk_0: pl.Tensor[[256, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 22)] = pl.tensor.view(wq_a_0, [256, 128], [k0_7 + AIV_IDX * 256, q0_0])
                pl.comm.tpush_to_aic(wq_chunk_0, AIV_IDX)
                _t7: pl.Tensor[[4, 256], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 24)] = pl.tensor.cast(normed_0, target_type=pl.BFLOAT16, mode=2)
                pl.comm.tpush_to_aic(_t7, AIV_IDX)
                _t8: pl.Tensor[[2, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 26)] = pl.comm.tpop_from_aic(AIV_IDX)
                q_acc_4: pl.Tensor[[2, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 28)] = pl.tensor.add(q_acc_iter_2, _t8)
                pl.comm.tfree_to_aic(AIV_IDX)
                k0_6, q_acc_3, x_chunk_6 = pl.yield_(k0_7, q_acc_4, x_chunk_7)
            _t9: pl.Tensor[[2, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 32)] = pl.tensor.cast(q_acc_3, target_type=pl.BFLOAT16, mode=2)
            qr_5: pl.Tensor[[16, 1536], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 49152, 34)] = pl.tensor.assemble(qr_iter_3_outer_l1, _t9, [b0_0 + AIV_IDX * 2, q0_0])
            k0_iter_3_outer_l1_rv, kb_iter_3_outer_l1_rv, qr_iter_3_outer_l1_rv, x_chunk_iter_3_outer_l1_rv = pl.yield_(k0_6, kb_5, qr_5, x_chunk_6)
        return k0_iter_3_outer_l1_rv, kb_iter_3_outer_l1_rv, qr_iter_3_outer_l1_rv, x_chunk_iter_3_outer_l1_rv
    @pl.function_group(aic="deepseek_v3_2_decode_front_layer_incore_0_aic", aiv="deepseek_v3_2_decode_front_layer_incore_0_aiv", aiv_runtime_params=["AIV_IDX"])
    class deepseek_v3_2_decode_front_layer_incore_0_group:
        """Parameter passing:
          call_group(deepseek_v3_2_decode_front_layer_incore_0_group, b0_0, hidden_states_0, input_rms_weight_0, inv_rms_tile_1, k0_0, k0_iter_1, k0_iter_3_outer_l0, kb_0, kb_iter_1, kb_iter_3_outer_l0, ob_0_out, qr_0, qr_iter_1, qr_iter_3_outer_l0, wq_a_0, x_chunk_0, x_chunk_iter_1, x_chunk_iter_3_outer_l0)
            → deepseek_v3_2_decode_front_layer_incore_0_aic(b0_0, hidden_states_0, input_rms_weight_0, inv_rms_tile_1, k0_0, k0_iter_1, k0_iter_3_outer_l0, kb_0, kb_iter_1, kb_iter_3_outer_l0, ob_0_out, qr_0, qr_iter_1, qr_iter_3_outer_l0, wq_a_0, x_chunk_0, x_chunk_iter_1, x_chunk_iter_3_outer_l0)
            → deepseek_v3_2_decode_front_layer_incore_0_aiv(b0_0, hidden_states_0, input_rms_weight_0, inv_rms_tile_1, k0_0, k0_iter_1, k0_iter_3_outer_l0, kb_0, kb_iter_1, kb_iter_3_outer_l0, ob_0_out, qr_0, qr_iter_1, qr_iter_3_outer_l0, wq_a_0, x_chunk_0, x_chunk_iter_1, x_chunk_iter_3_outer_l0, AIV_IDX=<runtime>)
        """
        pass

    @pl.function(type=pl.FunctionType.InCore)
    def deepseek_v3_2_decode_front_layer_incore_1_aic(self, b0_0: pl.Scalar[pl.INDEX], gamma_0: pl.Tensor[[1, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 0)], gamma_iter_1_outer_l0: pl.Tensor[[1, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 1)], k0_4: pl.Scalar[pl.INDEX], k0_iter_8_outer_l0: pl.Scalar[pl.INDEX], kb_4: pl.Scalar[pl.INDEX], kb_iter_6_outer_l0: pl.Scalar[pl.INDEX], ob_1_out: pl.Scalar[pl.INDEX], q0_0: pl.Scalar[pl.INDEX], q0_iter_1_outer_l0: pl.Scalar[pl.INDEX], q_acc_3: pl.Tensor[[4, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 2)], q_acc_iter_5_outer_l0: pl.Tensor[[4, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 3)], q_norm_weight_0: pl.Tensor[[1, 1536], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 6144, 4)], q_proj_0: pl.Tensor[[16, 128 * 192], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 5)], q_proj_iter_1: pl.Tensor[[16, 128 * 192], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 6)], q_proj_iter_3_outer_l0: pl.Tensor[[16, 128 * 192], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 7)], qr_iter_3_outer_l0_rv: pl.Tensor[[16, 1536], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 49152, 8)], wq_b_0: pl.Tensor[[1536, 24576], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 75497472, 9)], wq_chunk_0: pl.Tensor[[512, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 131072, 10)], wq_chunk_iter_1_outer_l0: pl.Tensor[[512, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 131072, 11)]) -> tuple[pl.Tensor[[1, 512], pl.FP32], pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Tensor[[4, 128], pl.FP32], pl.Tensor[[16, 128 * 192], pl.BFLOAT16], pl.Tensor[[512, 128], pl.BFLOAT16]]:
        pl.comm.aic_initialize_pipe()
        for ob_1_in, (gamma_iter_1_outer_l1, k0_iter_8_outer_l1, kb_iter_6_outer_l1, q0_iter_1_outer_l1, q_acc_iter_5_outer_l1, q_proj_iter_3_outer_l1, wq_chunk_iter_1_outer_l1) in pl.parallel(0, 8, 1, init_values=(gamma_iter_1_outer_l0, k0_iter_8_outer_l0, kb_iter_6_outer_l0, q0_iter_1_outer_l0, q_acc_iter_5_outer_l0, q_proj_iter_3_outer_l0, wq_chunk_iter_1_outer_l0)):
            q0_3: pl.Scalar[pl.INDEX] = (0 + (ob_1_out * 8 + ob_1_in) * 1) * 512
            for kb_8, (gamma_iter_3, k0_iter_10, wq_chunk_iter_3) in pl.range(0, 12, 1, init_values=(gamma_iter_1_outer_l1, k0_iter_8_outer_l1, wq_chunk_iter_1_outer_l1)):
                k0_12: pl.Scalar[pl.INDEX] = kb_8 * 128
                wq_chunk_5__h0: pl.Tensor[[64, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 12)] = pl.comm.tpop_from_aiv(0)
                wq_chunk_5__h1: pl.Tensor[[64, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 13)] = pl.comm.tpop_from_aiv(1)
                wq_chunk_5__tmp: pl.Tensor[[128, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 131072, 14)] = pl.tensor.create(__list__(128, 512), dtype=pl.BFLOAT16)
                wq_chunk_5__mid: pl.Tensor[[128, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 131072, 15)] = pl.tensor.assemble(wq_chunk_5__tmp, wq_chunk_5__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                wq_chunk_5: pl.Tensor[[128, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 131072, 16)] = pl.tensor.assemble(wq_chunk_5__mid, wq_chunk_5__h1, __list__(64, 0))
                pl.comm.tfree_to_aiv(1)
                _t11__h0: pl.Tensor[[2, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 17)] = pl.comm.tpop_from_aiv(0)
                _t11__h1: pl.Tensor[[2, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 18)] = pl.comm.tpop_from_aiv(1)
                _t11__tmp: pl.Tensor[[4, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 19)] = pl.tensor.create(__list__(4, 128), dtype=pl.BFLOAT16)
                _t11__mid: pl.Tensor[[4, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 20)] = pl.tensor.assemble(_t11__tmp, _t11__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                _t11: pl.Tensor[[4, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 21)] = pl.tensor.assemble(_t11__mid, _t11__h1, __list__(2, 0))
                pl.comm.tfree_to_aiv(1)
                _t12: pl.Tensor[[4, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 22)] = pl.tensor.matmul(_t11, wq_chunk_5, a_trans=False, b_trans=False, c_matrix_nz=False)
                __half0__: pl.Tensor[[2, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 23)] = pl.tensor.view(_t12, __list__(2, 512), __list__(0, 0))
                __half1__: pl.Tensor[[2, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 24)] = pl.tensor.view(_t12, __list__(2, 512), __list__(2, 0))
                pl.comm.tpush_to_aiv(__half0__, 0)
                pl.comm.tpush_to_aiv(__half1__, 1)
                gamma_4, k0_11, wq_chunk_4 = pl.yield_(k0_12, wq_chunk_5)
            gamma_iter_1_outer_l1_rv, k0_iter_8_outer_l1_rv, kb_iter_6_outer_l1_rv, q0_iter_1_outer_l1_rv, q_acc_iter_5_outer_l1_rv, q_proj_iter_3_outer_l1_rv, wq_chunk_iter_1_outer_l1_rv = pl.yield_(gamma_4, k0_11, kb_8, q0_3, wq_chunk_4)
        return gamma_iter_1_outer_l1_rv, k0_iter_8_outer_l1_rv, kb_iter_6_outer_l1_rv, q0_iter_1_outer_l1_rv, q_acc_iter_5_outer_l1_rv, q_proj_iter_3_outer_l1_rv, wq_chunk_iter_1_outer_l1_rv
    @pl.function(type=pl.FunctionType.InCore)
    def deepseek_v3_2_decode_front_layer_incore_1_aiv(self, b0_0: pl.Scalar[pl.INDEX], gamma_0: pl.Tensor[[1, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 0)], gamma_iter_1_outer_l0: pl.Tensor[[1, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 1)], k0_4: pl.Scalar[pl.INDEX], k0_iter_8_outer_l0: pl.Scalar[pl.INDEX], kb_4: pl.Scalar[pl.INDEX], kb_iter_6_outer_l0: pl.Scalar[pl.INDEX], ob_1_out: pl.Scalar[pl.INDEX], q0_0: pl.Scalar[pl.INDEX], q0_iter_1_outer_l0: pl.Scalar[pl.INDEX], q_acc_3: pl.Tensor[[4, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 2)], q_acc_iter_5_outer_l0: pl.Tensor[[4, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 3)], q_norm_weight_0: pl.Tensor[[1, 1536], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 6144, 4)], q_proj_0: pl.Tensor[[16, 128 * 192], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 5)], q_proj_iter_1: pl.Tensor[[16, 128 * 192], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 6)], q_proj_iter_3_outer_l0: pl.Tensor[[16, 128 * 192], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 7)], qr_iter_3_outer_l0_rv: pl.Tensor[[16, 1536], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 49152, 8)], wq_b_0: pl.Tensor[[1536, 24576], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 75497472, 9)], wq_chunk_0: pl.Tensor[[512, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 131072, 10)], wq_chunk_iter_1_outer_l0: pl.Tensor[[512, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 131072, 11)], AIV_IDX: pl.Scalar[pl.INDEX]) -> tuple[pl.Tensor[[1, 512], pl.FP32], pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Tensor[[4, 128], pl.FP32], pl.Tensor[[16, 128 * 192], pl.BFLOAT16], pl.Tensor[[512, 128], pl.BFLOAT16]]:
        pl.comm.aiv_initialize_pipe()
        for ob_1_in, (gamma_iter_1_outer_l1, k0_iter_8_outer_l1, kb_iter_6_outer_l1, q0_iter_1_outer_l1, q_acc_iter_5_outer_l1, q_proj_iter_3_outer_l1, wq_chunk_iter_1_outer_l1) in pl.parallel(0, 8, 1, init_values=(gamma_iter_1_outer_l0, k0_iter_8_outer_l0, kb_iter_6_outer_l0, q0_iter_1_outer_l0, q_acc_iter_5_outer_l0, q_proj_iter_3_outer_l0, wq_chunk_iter_1_outer_l0)):
            q0_3: pl.Scalar[pl.INDEX] = (0 + (ob_1_out * 8 + ob_1_in) * 1) * 512
            q_acc_7: pl.Tensor[[2, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 12)] = pl.tensor.create([2, 512], dtype=pl.FP32)
            q_acc_8: pl.Tensor[[4, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 14)] = pl.tensor.mul(q_acc_7, 0.0)
            for kb_8, (gamma_iter_3, k0_iter_10, q_acc_iter_9, wq_chunk_iter_3) in pl.range(0, 12, 1, init_values=(gamma_iter_1_outer_l1, k0_iter_8_outer_l1, q_acc_8, wq_chunk_iter_1_outer_l1)):
                k0_12: pl.Scalar[pl.INDEX] = kb_8 * 128
                _t10: pl.Tensor[[4, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 15)] = pl.tensor.view(qr_iter_3_outer_l0_rv, [4, 64], [b0_0, k0_12 + AIV_IDX * 64])
                q_chunk_0: pl.Tensor[[4, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 17)] = pl.tensor.cast(_t10, target_type=pl.FP32, mode=2)
                gamma_5: pl.Tensor[[1, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 18)] = pl.tensor.view(q_norm_weight_0, [1, 64], [0, k0_12 + AIV_IDX * 64])
                qn_0: pl.Tensor[[4, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 21)] = pl.tensor.col_expand_mul(q_chunk_0, gamma_5)
                wq_chunk_5: pl.Tensor[[64, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 22)] = pl.tensor.view(wq_b_0, [64, 512], [k0_12 + AIV_IDX * 64, q0_3])
                pl.comm.tpush_to_aic(wq_chunk_5, AIV_IDX)
                _t11: pl.Tensor[[4, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 24)] = pl.tensor.cast(qn_0, target_type=pl.BFLOAT16, mode=2)
                pl.comm.tpush_to_aic(_t11, AIV_IDX)
                _t12: pl.Tensor[[2, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 26)] = pl.comm.tpop_from_aic(AIV_IDX)
                q_acc_11: pl.Tensor[[2, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 28)] = pl.tensor.add(q_acc_iter_9, _t12)
                pl.comm.tfree_to_aic(AIV_IDX)
                gamma_4, k0_11, q_acc_10, wq_chunk_4 = pl.yield_(gamma_5, k0_12, q_acc_11, wq_chunk_5)
            _t13: pl.Tensor[[2, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 33)] = pl.tensor.cast(q_acc_10, target_type=pl.BFLOAT16, mode=2)
            q_proj_5: pl.Tensor[[16, 128 * 192], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 35)] = pl.tensor.assemble(q_proj_iter_3_outer_l1, _t13, [b0_0 + AIV_IDX * 2, q0_3])
            gamma_iter_1_outer_l1_rv, k0_iter_8_outer_l1_rv, kb_iter_6_outer_l1_rv, q0_iter_1_outer_l1_rv, q_acc_iter_5_outer_l1_rv, q_proj_iter_3_outer_l1_rv, wq_chunk_iter_1_outer_l1_rv = pl.yield_(gamma_4, k0_11, kb_8, q0_3, q_acc_10, q_proj_5, wq_chunk_4)
        return gamma_iter_1_outer_l1_rv, k0_iter_8_outer_l1_rv, kb_iter_6_outer_l1_rv, q0_iter_1_outer_l1_rv, q_acc_iter_5_outer_l1_rv, q_proj_iter_3_outer_l1_rv, wq_chunk_iter_1_outer_l1_rv
    @pl.function_group(aic="deepseek_v3_2_decode_front_layer_incore_1_aic", aiv="deepseek_v3_2_decode_front_layer_incore_1_aiv", aiv_runtime_params=["AIV_IDX"])
    class deepseek_v3_2_decode_front_layer_incore_1_group:
        """Parameter passing:
          call_group(deepseek_v3_2_decode_front_layer_incore_1_group, b0_0, gamma_0, gamma_iter_1_outer_l0, k0_4, k0_iter_8_outer_l0, kb_4, kb_iter_6_outer_l0, ob_1_out, q0_0, q0_iter_1_outer_l0, q_acc_3, q_acc_iter_5_outer_l0, q_norm_weight_0, q_proj_0, q_proj_iter_1, q_proj_iter_3_outer_l0, qr_iter_3_outer_l0_rv, wq_b_0, wq_chunk_0, wq_chunk_iter_1_outer_l0)
            → deepseek_v3_2_decode_front_layer_incore_1_aic(b0_0, gamma_0, gamma_iter_1_outer_l0, k0_4, k0_iter_8_outer_l0, kb_4, kb_iter_6_outer_l0, ob_1_out, q0_0, q0_iter_1_outer_l0, q_acc_3, q_acc_iter_5_outer_l0, q_norm_weight_0, q_proj_0, q_proj_iter_1, q_proj_iter_3_outer_l0, qr_iter_3_outer_l0_rv, wq_b_0, wq_chunk_0, wq_chunk_iter_1_outer_l0)
            → deepseek_v3_2_decode_front_layer_incore_1_aiv(b0_0, gamma_0, gamma_iter_1_outer_l0, k0_4, k0_iter_8_outer_l0, kb_4, kb_iter_6_outer_l0, ob_1_out, q0_0, q0_iter_1_outer_l0, q_acc_3, q_acc_iter_5_outer_l0, q_norm_weight_0, q_proj_0, q_proj_iter_1, q_proj_iter_3_outer_l0, qr_iter_3_outer_l0_rv, wq_b_0, wq_chunk_0, wq_chunk_iter_1_outer_l0, AIV_IDX=<runtime>)
        """
        pass

    @pl.function(type=pl.FunctionType.InCore)
    def deepseek_v3_2_decode_front_layer_incore_2_aic(self, attn_row_1: pl.Tensor[[1, 16384], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 0)], attn_row_iter_2_outer_l0: pl.Tensor[[1, 16384], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 1)], b_0: pl.Scalar[pl.INDEX], cos_hi_0: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 2)], cos_lo_0: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 3)], ctx_len_0: pl.Scalar[pl.INT32], h_0_out: pl.Scalar[pl.INDEX], kk_8: pl.Scalar[pl.INDEX], kk_iter_10_outer_l0: pl.Scalar[pl.INDEX], kv_cache_3: pl.Tensor[[65536, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 67108864, 4)], pe_cache_3: pl.Tensor[[65536, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 8388608, 5)], q_proj_2: pl.Tensor[[16, 128 * 192], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 6)], s_0: pl.Scalar[pl.INDEX], s_iter_1_outer_l0: pl.Scalar[pl.INDEX], sin_hi_0: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 7)], sin_lo_0: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 8)], topk_idx_6: pl.Tensor[[1, 2048], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 9)], w_latent_to_v_0: pl.Tensor[[128, 512, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16777216, 10)], w_q_nope_to_latent_0: pl.Tensor[[128, 128, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16777216, 11)]) -> tuple[pl.Tensor[[1, 16384], pl.FP32], pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX]]:
        pl.comm.aic_initialize_pipe()
        for h_0_in, (attn_row_iter_2_outer_l1, kk_iter_10_outer_l1, s_iter_1_outer_l1) in pl.parallel(0, 8, 1, init_values=(attn_row_iter_2_outer_l0, kk_iter_10_outer_l0, s_iter_1_outer_l0)):
            q_col_0: pl.Scalar[pl.INDEX] = (0 + (h_0_out * 8 + h_0_in) * 1) * 192
            _t51: pl.Tensor[[1, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 12)] = pl.comm.tpop_from_aiv()
            _t52__h0: pl.Tensor[[64, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 13)] = pl.comm.tpop_from_aiv(0)
            _t52__h1: pl.Tensor[[64, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 14)] = pl.comm.tpop_from_aiv(1)
            _t52__tmp: pl.Tensor[[128, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 131072, 15)] = pl.tensor.create(__list__(128, 512), dtype=pl.BFLOAT16)
            _t52__mid: pl.Tensor[[128, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 131072, 16)] = pl.tensor.assemble(_t52__tmp, _t52__h0, __list__(0, 0))
            pl.comm.tfree_to_aiv(0)
            _t52: pl.Tensor[[128, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 131072, 17)] = pl.tensor.assemble(_t52__mid, _t52__h1, __list__(64, 0))
            pl.comm.tfree_to_aiv(1)
            q_nope_latent_0: pl.Tensor[[1, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 18)] = pl.tensor.matmul(_t51, _t52, a_trans=False, b_trans=False, c_matrix_nz=False)
            pl.comm.tfree_to_aiv()
            pl.comm.tpush_to_aiv(q_nope_latent_0, 0)
            pl.comm.tpush_to_aiv(q_nope_latent_0, 1)
            sparse_k_0: pl.Scalar[pl.INDEX] = min(2048, pl.cast(ctx_len_0, pl.INDEX))
            for kk_12, (s_iter_3,) in pl.range(0, sparse_k_0, 1, init_values=(s_iter_1_outer_l1,)):
                s_5: pl.Scalar[pl.INDEX] = pl.tensor.read(topk_idx_6, [0, kk_12])
                if s_5 >= 0:
                    cache_s_1: pl.Scalar[pl.INDEX] = b_0 * 4096 + s_5
                s_4: pl.Scalar[pl.INDEX] = pl.yield_(li_7, mi_7, oi_7, s_5)
            v_col_0: pl.Scalar[pl.INDEX] = (0 + (h_0_out * 8 + h_0_in) * 1) * 128
            for vb_0 in pl.range(0, 2, 1):
                v0_0: pl.Scalar[pl.INDEX] = vb_0 * 64
                wv_tile_0__h0: pl.Tensor[[256, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 22)] = pl.comm.tpop_from_aiv(0)
                wv_tile_0__h1: pl.Tensor[[256, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 23)] = pl.comm.tpop_from_aiv(1)
                wv_tile_0__tmp: pl.Tensor[[512, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 24)] = pl.tensor.create(__list__(512, 64), dtype=pl.BFLOAT16)
                wv_tile_0__mid: pl.Tensor[[512, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 25)] = pl.tensor.assemble(wv_tile_0__tmp, wv_tile_0__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                wv_tile_0: pl.Tensor[[512, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 26)] = pl.tensor.assemble(wv_tile_0__mid, wv_tile_0__h1, __list__(256, 0))
                pl.comm.tfree_to_aiv(1)
                _t65: pl.Tensor[[1, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 27)] = pl.comm.tpop_from_aiv()
                v_part_0: pl.Tensor[[1, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 28)] = pl.tensor.matmul(_t65, wv_tile_0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                pl.comm.tfree_to_aiv()
                pl.comm.tpush_to_aiv(v_part_0)
            attn_row_iter_2_outer_l1_rv, kk_iter_10_outer_l1_rv, s_iter_1_outer_l1_rv = pl.yield_(kk_12, s_4)
        return attn_row_iter_2_outer_l1_rv, kk_iter_10_outer_l1_rv, s_iter_1_outer_l1_rv
    @pl.function(type=pl.FunctionType.InCore)
    def deepseek_v3_2_decode_front_layer_incore_2_aiv(self, attn_row_1: pl.Tensor[[1, 16384], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 0)], attn_row_iter_2_outer_l0: pl.Tensor[[1, 16384], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 1)], b_0: pl.Scalar[pl.INDEX], cos_hi_0: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 2)], cos_lo_0: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 3)], ctx_len_0: pl.Scalar[pl.INT32], h_0_out: pl.Scalar[pl.INDEX], kk_8: pl.Scalar[pl.INDEX], kk_iter_10_outer_l0: pl.Scalar[pl.INDEX], kv_cache_3: pl.Tensor[[65536, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 67108864, 4)], pe_cache_3: pl.Tensor[[65536, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 8388608, 5)], q_proj_2: pl.Tensor[[16, 128 * 192], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 6)], s_0: pl.Scalar[pl.INDEX], s_iter_1_outer_l0: pl.Scalar[pl.INDEX], sin_hi_0: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 7)], sin_lo_0: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 8)], topk_idx_6: pl.Tensor[[1, 2048], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 9)], w_latent_to_v_0: pl.Tensor[[128, 512, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16777216, 10)], w_q_nope_to_latent_0: pl.Tensor[[128, 128, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16777216, 11)], AIV_IDX: pl.Scalar[pl.INDEX]) -> tuple[pl.Tensor[[1, 16384], pl.FP32], pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX]]:
        mem_vec_12: pl.MemRefType = pl.block.alloc(pl.MemorySpace.Vec, -1, 0, 12)
        mem_vec_13: pl.MemRefType = pl.block.alloc(pl.MemorySpace.Vec, -1, 0, 13)
        mem_vec_14: pl.MemRefType = pl.block.alloc(pl.MemorySpace.Vec, -1, 0, 14)
        mem_vec_15: pl.MemRefType = pl.block.alloc(pl.MemorySpace.Vec, -1, 0, 15)
        cos_hi_0_tile: pl.Tile[[1, 64 // 2], pl.FP32, tile_view=pl.TileView(valid_shape=[1, 64 // 2], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null), pl.MemRef(pl.MemorySpace.Vec, -1, 0, 12)] = pl.block.load(cos_hi_0, [0, 0], [1, 64 // 2], [1, 64 // 2], target_memory=pl.MemorySpace.Vec)
        cos_lo_0_tile: pl.Tile[[1, 64 // 2], pl.FP32, tile_view=pl.TileView(valid_shape=[1, 64 // 2], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null), pl.MemRef(pl.MemorySpace.Vec, -1, 0, 13)] = pl.block.load(cos_lo_0, [0, 0], [1, 64 // 2], [1, 64 // 2], target_memory=pl.MemorySpace.Vec)
        sin_hi_0_tile: pl.Tile[[1, 64 // 2], pl.FP32, tile_view=pl.TileView(valid_shape=[1, 64 // 2], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null), pl.MemRef(pl.MemorySpace.Vec, -1, 0, 14)] = pl.block.load(sin_hi_0, [0, 0], [1, 64 // 2], [1, 64 // 2], target_memory=pl.MemorySpace.Vec)
        sin_lo_0_tile: pl.Tile[[1, 64 // 2], pl.FP32, tile_view=pl.TileView(valid_shape=[1, 64 // 2], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null), pl.MemRef(pl.MemorySpace.Vec, -1, 0, 15)] = pl.block.load(sin_lo_0, [0, 0], [1, 64 // 2], [1, 64 // 2], target_memory=pl.MemorySpace.Vec)
        pl.comm.aiv_initialize_pipe()
        for h_0_in, (attn_row_iter_2_outer_l1, kk_iter_10_outer_l1, s_iter_1_outer_l1) in pl.parallel(0, 8, 1, init_values=(attn_row_iter_2_outer_l0, kk_iter_10_outer_l0, s_iter_1_outer_l0)):
            q_col_0: pl.Scalar[pl.INDEX] = (0 + (h_0_out * 8 + h_0_in) * 1) * 192
            _t43: pl.Tensor[[1, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 128, 16)] = pl.tensor.view(q_proj_2, [1, 64], [b_0, q_col_0 + AIV_IDX * 64])
            q_nope_0: pl.Tensor[[1, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 18)] = pl.tensor.cast(_t43, target_type=pl.FP32, mode=2)
            _t44: pl.Tensor[[1, 32], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 19)] = pl.tensor.view(q_proj_2, [1, 32], [b_0, q_col_0 + 128 + AIV_IDX * 32])
            q_pe_0: pl.Tensor[[1, 32], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 128, 21)] = pl.tensor.cast(_t44, target_type=pl.FP32, mode=2)
            q_lo_0: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 23)] = pl.tensor.deep_view(q_pe_0, [1, 64 // 2], [0, 0])
            q_hi_0: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 24)] = pl.tensor.deep_view(q_pe_0, [1, 64 // 2], [0, 64 // 2])
            q_rot_0: pl.Tensor[[1, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 25)] = pl.tensor.create([1, 64], dtype=pl.FP32)
            _t45: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 26)] = pl.tensor.col_expand_mul(q_lo_0, cos_lo_0)
            _t46: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 27)] = pl.tensor.col_expand_mul(q_hi_0, sin_lo_0)
            _t47: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 28)] = pl.tensor.sub(_t45, _t46)
            q_rot_1: pl.Tensor[[1, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 29)] = pl.tensor.assemble(q_rot_0, _t47, [0, 0])
            _t48: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 30)] = pl.tensor.col_expand_mul(q_hi_0, cos_hi_0)
            _t49: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 31)] = pl.tensor.col_expand_mul(q_lo_0, sin_hi_0)
            _t50: pl.Tensor[[1, 64 // 2], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 32)] = pl.tensor.add(_t48, _t49)
            q_rot_2: pl.Tensor[[1, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 33)] = pl.tensor.assemble(q_rot_1, _t50, [0, 64 // 2])
            _t51: pl.Tensor[[1, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 128, 35)] = pl.tensor.cast(q_nope_0, target_type=pl.BFLOAT16, mode=2)
            pl.comm.tpush_to_aic(_t51, AIV_IDX)
            _t52: pl.Tensor[[64, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 37)] = pl.tensor.view(w_q_nope_to_latent_0, [64, 512], [0 + (h_0_out * 8 + h_0_in) * 1 + AIV_IDX * 64, 0, 0])
            pl.comm.tpush_to_aic(_t52, AIV_IDX)
            q_nope_latent_0: pl.Tensor[[1, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 39)] = pl.comm.tpop_from_aic()
            oi_0: pl.Tensor[[1, 256], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 40)] = pl.tensor.create([1, 256], dtype=pl.FP32)
            li_0: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 41)] = pl.tensor.create([1, 1], dtype=pl.FP32)
            mi_0: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 42)] = pl.tensor.create([1, 1], dtype=pl.FP32)
            oi_1: pl.Tensor[[1, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 44)] = pl.tensor.mul(oi_0, 0.0)
            li_1: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 45)] = pl.tensor.mul(li_0, 0.0)
            mi_1: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 46)] = pl.tensor.mul(mi_0, 0.0)
            sparse_k_0: pl.Scalar[pl.INDEX] = min(2048, pl.cast(ctx_len_0, pl.INDEX))
            for kk_12, (li_iter_2, mi_iter_2, oi_iter_2, s_iter_3) in pl.range(0, sparse_k_0, 1, init_values=(li_1, mi_1, oi_1, s_iter_1_outer_l1)):
                s_5: pl.Scalar[pl.INDEX] = pl.tensor.read(topk_idx_6, [0, kk_12])
                if s_5 >= 0:
                    cache_s_1: pl.Scalar[pl.INDEX] = b_0 * 4096 + s_5
                    _t53: pl.Tensor[[1, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 47)] = pl.tensor.view(kv_cache_3, [1, 512], [cache_s_1, 0])
                    kv_s_1: pl.Tensor[[1, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 48)] = pl.tensor.cast(_t53, target_type=pl.FP32, mode=2)
                    _t54: pl.Tensor[[1, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 128, 49)] = pl.tensor.view(pe_cache_3, [1, 64], [cache_s_1, 0])
                    pe_s_1: pl.Tensor[[1, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 50)] = pl.tensor.cast(_t54, target_type=pl.FP32, mode=2)
                    _t55: pl.Tensor[[1, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 51)] = pl.tensor.mul(q_nope_latent_0, kv_s_1)
                    score_nope_1: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 52)] = pl.tensor.row_sum(_t55)
                    _t56: pl.Tensor[[1, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 53)] = pl.tensor.mul(q_rot_2, pe_s_1)
                    score_pe_1: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 54)] = pl.tensor.row_sum(_t56)
                    _t57: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 55)] = pl.tensor.add(score_nope_1, score_pe_1)
                    score_0: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 56)] = pl.tensor.mul(_t57, 0.0721688)
                    cur_mi_0: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 57)] = score_0
                    _t58: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 58)] = pl.tensor.sub(score_0, cur_mi_0)
                    cur_li_0: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 59)] = pl.tensor.exp(_t58)
                    oi_tmp_0: pl.Tensor[[1, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 60)] = pl.tensor.row_expand_mul(kv_s_1, cur_li_0)
                    if kk_12 == 0:
                        oi_4: pl.Tensor[[1, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 61)] = oi_tmp_0
                        li_4: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 62)] = cur_li_0
                        mi_4: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 63)] = cur_mi_0
                        li_6, mi_6, oi_6 = pl.yield_(li_4, mi_4, oi_4)
                    else:
                        mi_new_0: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 64)] = pl.tensor.maximum(mi_iter_2, cur_mi_0)
                        _t59: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 65)] = pl.tensor.sub(mi_iter_2, mi_new_0)
                        alpha_0: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 66)] = pl.tensor.exp(_t59)
                        _t60: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 67)] = pl.tensor.sub(cur_mi_0, mi_new_0)
                        beta_0: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 68)] = pl.tensor.exp(_t60)
                        _t61: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 69)] = pl.tensor.mul(alpha_0, li_iter_2)
                        _t62: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 70)] = pl.tensor.mul(beta_0, cur_li_0)
                        li_5: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 71)] = pl.tensor.add(_t61, _t62)
                        _t63: pl.Tensor[[1, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 72)] = pl.tensor.row_expand_mul(oi_iter_2, alpha_0)
                        _t64: pl.Tensor[[1, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 73)] = pl.tensor.row_expand_mul(oi_tmp_0, beta_0)
                        oi_5: pl.Tensor[[1, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 74)] = pl.tensor.add(_t63, _t64)
                        mi_5: pl.Tensor[[1, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 75)] = mi_new_0
                        li_6, mi_6, oi_6 = pl.yield_(li_5, mi_5, oi_5)
                    li_7, mi_7, oi_7 = pl.yield_(li_6, mi_6, oi_6)
                else:
                    li_7, mi_7, oi_7 = pl.yield_(li_iter_2, mi_iter_2, oi_iter_2)
                li_3, mi_3, oi_3, s_4 = pl.yield_(li_7, mi_7, oi_7, s_5)
            pl.comm.tfree_to_aic()
            ctx_latent_0: pl.Tensor[[1, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 85)] = pl.tensor.row_expand_div(oi_3, li_3)
            v_col_0: pl.Scalar[pl.INDEX] = (0 + (h_0_out * 8 + h_0_in) * 1) * 128
            ctx_v_0: pl.Tensor[[1, 64], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 86)] = pl.tensor.create([1, 64], dtype=pl.FP32)
            ctx_v_1: pl.Tensor[[1, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 88)] = pl.tensor.mul(ctx_v_0, 0.0)
            for vb_0, (ctx_v_iter_2,) in pl.range(0, 2, 1, init_values=(ctx_v_1,)):
                v0_0: pl.Scalar[pl.INDEX] = vb_0 * 64
                wv_tile_0: pl.Tensor[[256, 64], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 89)] = pl.tensor.view(w_latent_to_v_0, [256, 64], [0 + (h_0_out * 8 + h_0_in) * 1 + AIV_IDX * 256, 0, v0_0])
                pl.comm.tpush_to_aic(wv_tile_0, AIV_IDX)
                _t65: pl.Tensor[[1, 256], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 91)] = pl.tensor.cast(ctx_latent_0, target_type=pl.BFLOAT16, mode=2)
                pl.comm.tpush_to_aic(_t65, AIV_IDX)
                v_part_0: pl.Tensor[[1, 32], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 128, 93)] = pl.comm.tpop_from_aic(AIV_IDX)
                ctx_v_4: pl.Tensor[[1, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 95)] = pl.tensor.assemble(ctx_v_iter_2, v_part_0, [0, v0_0 + AIV_IDX * 32])
                pl.comm.tfree_to_aic(AIV_IDX)
                ctx_v_3: pl.Tensor[[1, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 96)] = pl.yield_(ctx_v_4)
            attn_row_4: pl.Tensor[[1, 16384], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 97)] = pl.tensor.assemble(attn_row_iter_2_outer_l1, ctx_v_3, [0, v_col_0])
            attn_row_iter_2_outer_l1_rv, kk_iter_10_outer_l1_rv, s_iter_1_outer_l1_rv = pl.yield_(attn_row_4, kk_12, s_4)
        return attn_row_iter_2_outer_l1_rv, kk_iter_10_outer_l1_rv, s_iter_1_outer_l1_rv
    @pl.function_group(aic="deepseek_v3_2_decode_front_layer_incore_2_aic", aiv="deepseek_v3_2_decode_front_layer_incore_2_aiv", aiv_runtime_params=["AIV_IDX"])
    class deepseek_v3_2_decode_front_layer_incore_2_group:
        """Parameter passing:
          call_group(deepseek_v3_2_decode_front_layer_incore_2_group, attn_row_1, attn_row_iter_2_outer_l0, b_0, cos_hi_0, cos_lo_0, ctx_len_0, h_0_out, kk_8, kk_iter_10_outer_l0, kv_cache_3, pe_cache_3, q_proj_2, s_0, s_iter_1_outer_l0, sin_hi_0, sin_lo_0, topk_idx_6, w_latent_to_v_0, w_q_nope_to_latent_0)
            → deepseek_v3_2_decode_front_layer_incore_2_aic(attn_row_1, attn_row_iter_2_outer_l0, b_0, cos_hi_0, cos_lo_0, ctx_len_0, h_0_out, kk_8, kk_iter_10_outer_l0, kv_cache_3, pe_cache_3, q_proj_2, s_0, s_iter_1_outer_l0, sin_hi_0, sin_lo_0, topk_idx_6, w_latent_to_v_0, w_q_nope_to_latent_0)
            → deepseek_v3_2_decode_front_layer_incore_2_aiv(attn_row_1, attn_row_iter_2_outer_l0, b_0, cos_hi_0, cos_lo_0, ctx_len_0, h_0_out, kk_8, kk_iter_10_outer_l0, kv_cache_3, pe_cache_3, q_proj_2, s_0, s_iter_1_outer_l0, sin_hi_0, sin_lo_0, topk_idx_6, w_latent_to_v_0, w_q_nope_to_latent_0, AIV_IDX=<runtime>)
        """
        pass
