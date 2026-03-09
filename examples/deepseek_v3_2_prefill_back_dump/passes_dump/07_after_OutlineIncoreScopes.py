# pypto.program: DeepSeekV32PrefillBack
import pypto.language as pl

@pl.program
class DeepSeekV32PrefillBack:
    @pl.function(type=pl.FunctionType.InCore)
    def deepseek_v3_2_prefill_back_layer_incore_0(self, b_0: pl.Scalar[pl.INDEX], combined_tile_0: pl.Tensor[[4, 16384], pl.FP32], hidden_states_0: pl.Tensor[[16, 4096, 7168], pl.BFLOAT16], ob_0_out: pl.Scalar[pl.INDEX], p0_0: pl.Scalar[pl.INDEX], resid1_tile_0: pl.Tensor[[4, 7168], pl.FP32], resid1_tile_iter_1_outer_l0: pl.Tensor[[4, 7168], pl.FP32], wo_0: pl.Tensor[[16384, 7168], pl.BFLOAT16]) -> pl.Tensor[[4, 7168], pl.FP32]:
        for ob_0_in, (resid1_tile_iter_1_outer_l1,) in pl.parallel(0, 8, 1, init_values=(resid1_tile_iter_1_outer_l0,)):
            o0_0: pl.Scalar[pl.INDEX] = (0 + (ob_0_out * 8 + ob_0_in) * 1) * 128
            o_acc_0: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.create([4, 128], dtype=pl.FP32)
            o_acc_1: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.mul(o_acc_0, 0.0)
            for kb_0, (o_acc_iter_2,) in pl.range(0, 32, 1, init_values=(o_acc_1,)):
                k0_0: pl.Scalar[pl.INDEX] = kb_0 * 512
                _t1: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.view(combined_tile_0, [4, 512], [0, k0_0])
                a_chunk_0: pl.Tensor[[4, 512], pl.BFLOAT16] = pl.tensor.cast(_t1, target_type=pl.BFLOAT16, mode=2)
                w_chunk_0: pl.Tensor[[512, 128], pl.BFLOAT16] = pl.tensor.view(wo_0, [512, 128], [k0_0, o0_0])
                _t2: pl.Tensor[[4, 128], pl.BFLOAT16] = pl.tensor.matmul(a_chunk_0, w_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                o_acc_4: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.add(o_acc_iter_2, _t2)
                o_acc_3: pl.Tensor[[4, 128], pl.FP32] = pl.yield_(o_acc_4)
            _t3: pl.Tensor[[4, 128], pl.BFLOAT16] = pl.tensor.view(hidden_states_0, [4, 128], [b_0, p0_0, o0_0])
            resid_0: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.cast(_t3, target_type=pl.FP32, mode=2)
            _t4: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.add(o_acc_3, resid_0)
            resid1_tile_3: pl.Tensor[[4, 7168], pl.FP32] = pl.tensor.assemble(resid1_tile_iter_1_outer_l1, _t4, [0, o0_0])
            resid1_tile_iter_1_outer_l1_rv: pl.Tensor[[4, 7168], pl.FP32] = pl.yield_(resid1_tile_3)
        return resid1_tile_iter_1_outer_l1_rv
    @pl.function(type=pl.FunctionType.InCore)
    def deepseek_v3_2_prefill_back_layer_incore_1(self, dob_0_out: pl.Scalar[pl.INDEX], down_proj_tile_1: pl.Tensor[[4, 7168], pl.FP32], down_proj_tile_iter_2: pl.Tensor[[4, 7168], pl.FP32], down_proj_tile_iter_4_outer_l0: pl.Tensor[[4, 7168], pl.FP32], mlp_chunk_bf16_0: pl.Tensor[[4, 512], pl.BFLOAT16], o0_3: pl.Scalar[pl.INDEX], w_down_0: pl.Tensor[[18432, 7168], pl.BFLOAT16]) -> pl.Tensor[[4, 7168], pl.FP32]:
        for dob_0_in, (down_proj_tile_iter_4_outer_l1,) in pl.parallel(0, 8, 1, init_values=(down_proj_tile_iter_4_outer_l0,)):
            d0_0: pl.Scalar[pl.INDEX] = (0 + (dob_0_out * 8 + dob_0_in) * 1) * 128
            down_prev_0: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.view(down_proj_tile_iter_4_outer_l1, [4, 128], [0, d0_0])
            w_down_chunk_0: pl.Tensor[[512, 128], pl.BFLOAT16] = pl.tensor.view(w_down_0, [512, 128], [o0_3, d0_0])
            _t17: pl.Tensor[[4, 128], pl.BFLOAT16] = pl.tensor.matmul(mlp_chunk_bf16_0, w_down_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
            down_next_0: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.add(down_prev_0, _t17)
            down_proj_tile_6: pl.Tensor[[4, 7168], pl.FP32] = pl.tensor.assemble(down_proj_tile_iter_4_outer_l1, down_next_0, [0, d0_0])
            down_proj_tile_iter_4_outer_l1_rv: pl.Tensor[[4, 7168], pl.FP32] = pl.yield_(down_proj_tile_6)
        return down_proj_tile_iter_4_outer_l1_rv
    @pl.function(type=pl.FunctionType.InCore)
    def deepseek_v3_2_prefill_back_layer_incore_2(self, b_0: pl.Scalar[pl.INDEX], down_proj_tile_3: pl.Tensor[[4, 7168], pl.FP32], o0_2: pl.Scalar[pl.INDEX], o0_iter_4_outer_l0: pl.Scalar[pl.INDEX], ob_2_out: pl.Scalar[pl.INDEX], out_0: pl.Tensor[[16, 4096, 7168], pl.BFLOAT16], out_iter_1: pl.Tensor[[16, 4096, 7168], pl.BFLOAT16], out_iter_3: pl.Tensor[[16, 4096, 7168], pl.BFLOAT16], out_iter_5_outer_l0: pl.Tensor[[16, 4096, 7168], pl.BFLOAT16], p0_0: pl.Scalar[pl.INDEX], resid1_tile_iter_1_outer_l0_rv: pl.Tensor[[4, 7168], pl.FP32]) -> tuple[pl.Scalar[pl.INDEX], pl.Tensor[[16, 4096, 7168], pl.BFLOAT16]]:
        for ob_2_in, (o0_iter_4_outer_l1, out_iter_5_outer_l1) in pl.parallel(0, 8, 1, init_values=(o0_iter_4_outer_l0, out_iter_5_outer_l0)):
            o0_6: pl.Scalar[pl.INDEX] = (0 + (ob_2_out * 8 + ob_2_in) * 1) * 128
            _t18: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.view(down_proj_tile_3, [4, 128], [0, o0_6])
            _t19: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.view(resid1_tile_iter_1_outer_l0_rv, [4, 128], [0, o0_6])
            down_acc_0: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.add(_t18, _t19)
            _t20: pl.Tensor[[4, 128], pl.BFLOAT16] = pl.tensor.cast(down_acc_0, target_type=pl.BFLOAT16, mode=2)
            out_7: pl.Tensor[[16, 4096, 7168], pl.BFLOAT16] = pl.tensor.assemble(out_iter_5_outer_l1, _t20, [b_0, p0_0, o0_6])
            o0_iter_4_outer_l1_rv, out_iter_5_outer_l1_rv = pl.yield_(o0_6, out_7)
        return o0_iter_4_outer_l1_rv, out_iter_5_outer_l1_rv
    @pl.function(type=pl.FunctionType.Orchestration)
    def deepseek_v3_2_prefill_back_layer(self, hidden_states_0: pl.Tensor[[16, 4096, 7168], pl.BFLOAT16], seq_lens_0: pl.Tensor[[16], pl.INT32], node_id_t_0: pl.Tensor[[1], pl.INT32], combine_buf_0: pl.Tensor[[128, 16, 4096, 16384], pl.BFLOAT16], wo_0: pl.Tensor[[16384, 7168], pl.BFLOAT16], post_rms_weight_0: pl.Tensor[[1, 7168], pl.FP32], w_gate_0: pl.Tensor[[7168, 18432], pl.BFLOAT16], w_up_0: pl.Tensor[[7168, 18432], pl.BFLOAT16], w_down_0: pl.Tensor[[18432, 7168], pl.BFLOAT16], out_0: pl.Tensor[[16, 4096, 7168], pl.BFLOAT16]) -> pl.Tensor[[16, 4096, 7168], pl.BFLOAT16]:
        node_id_0: pl.Scalar[pl.INT32] = pl.tensor.read(node_id_t_0, [0])
        for b_0, (out_iter_1,) in pl.parallel(0, 16, 1, init_values=(out_0,), chunk=4):
            seq_len_b_0: pl.Scalar[pl.INT32] = pl.tensor.read(seq_lens_0, [b_0])
            tok_blocks_0: pl.Scalar[pl.INDEX] = (pl.cast(seq_len_b_0, pl.INDEX) + 4 - 1) // 4
            for p0_idx_0, (out_iter_3,) in pl.range(0, tok_blocks_0, 1, init_values=(out_iter_1,)):
                p0_0: pl.Scalar[pl.INDEX] = p0_idx_0 * 4
                valid_tok_0: pl.Scalar[pl.INDEX] = min(4, pl.cast(seq_len_b_0, pl.INDEX) - p0_0)
                _t0: pl.Tensor[[4, 16384], pl.BFLOAT16] = pl.tensor.view(combine_buf_0, [4, 16384], [node_id_0, b_0, p0_0, 0])
                combined_tile_0: pl.Tensor[[4, 16384], pl.FP32] = pl.tensor.cast(_t0, target_type=pl.FP32, mode=2)
                resid1_tile_0: pl.Tensor[[4, 7168], pl.FP32] = pl.tensor.create([4, 7168], dtype=pl.FP32)
                for ob_0_out, (resid1_tile_iter_1_outer_l0,) in pl.range(0, 7, 1, init_values=(resid1_tile_0,)):
                    resid1_tile_iter_1_outer_l1_rv: pl.Tensor[[4, 7168], pl.FP32] = self.deepseek_v3_2_prefill_back_layer_incore_0(b_0, combined_tile_0, hidden_states_0, ob_0_out, p0_0, resid1_tile_0, resid1_tile_iter_1_outer_l0, wo_0)
                    resid1_tile_iter_1_outer_l0_rv: pl.Tensor[[4, 7168], pl.FP32] = pl.yield_(resid1_tile_iter_1_outer_l1_rv)
                sq_sum_0: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.create([4, 1], dtype=pl.FP32)
                sq_sum_1: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.mul(sq_sum_0, 0.0)
                for kb_1, (k0_iter_1, sq_sum_iter_2) in pl.range(0, 14, 1, init_values=(k0_0, sq_sum_1)):
                    k0_3: pl.Scalar[pl.INDEX] = kb_1 * 512
                    x_chunk_0: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.view(resid1_tile_iter_1_outer_l0_rv, [4, 512], [0, k0_3])
                    _t5: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.mul(x_chunk_0, x_chunk_0)
                    _t6: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.row_sum(_t5)
                    sq_sum_4: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.add(sq_sum_iter_2, _t6)
                    k0_2, sq_sum_3 = pl.yield_(k0_3, sq_sum_4)
                _t7: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.mul(sq_sum_3, 0.000139509)
                _t8: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.add(_t7, 1e-06)
                inv_rms_0: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.rsqrt(_t8)
                post_norm_tile_0: pl.Tensor[[4, 7168], pl.BFLOAT16] = pl.tensor.create([4, 7168], dtype=pl.BFLOAT16)
                down_proj_tile_0: pl.Tensor[[4, 7168], pl.FP32] = pl.tensor.create([4, 7168], dtype=pl.FP32)
                down_proj_tile_1: pl.Tensor[[4, 7168], pl.FP32] = pl.tensor.mul(down_proj_tile_0, 0.0)
                for kb_2, (k0_iter_4, post_norm_tile_iter_1, x_chunk_iter_1) in pl.range(0, 14, 1, init_values=(k0_2, post_norm_tile_0, x_chunk_0)):
                    k0_6: pl.Scalar[pl.INDEX] = kb_2 * 512
                    x_chunk_3: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.view(resid1_tile_iter_1_outer_l0_rv, [4, 512], [0, k0_6])
                    gamma_0: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.view(post_rms_weight_0, [1, 512], [0, k0_6])
                    _t9: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.row_expand_mul(x_chunk_3, inv_rms_0)
                    normed_0: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.col_expand_mul(_t9, gamma_0)
                    _t10: pl.Tensor[[4, 512], pl.BFLOAT16] = pl.tensor.cast(normed_0, target_type=pl.BFLOAT16, mode=2)
                    post_norm_tile_3: pl.Tensor[[4, 7168], pl.BFLOAT16] = pl.tensor.assemble(post_norm_tile_iter_1, _t10, [0, k0_6])
                    k0_5, post_norm_tile_2, x_chunk_2 = pl.yield_(k0_6, post_norm_tile_3, x_chunk_3)
                for ob_1, (down_proj_tile_iter_2, k0_iter_7, kb_iter_3, o0_iter_1) in pl.range(0, 36, 1, init_values=(down_proj_tile_1, k0_5, kb_2, o0_0)):
                    o0_3: pl.Scalar[pl.INDEX] = ob_1 * 512
                    gate_acc_0: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.create([4, 512], dtype=pl.FP32)
                    up_acc_0: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.create([4, 512], dtype=pl.FP32)
                    gate_acc_1: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.mul(gate_acc_0, 0.0)
                    up_acc_1: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.mul(up_acc_0, 0.0)
                    for kb_5, (gate_acc_iter_2, k0_iter_9, up_acc_iter_2) in pl.range(0, 14, 1, init_values=(gate_acc_1, k0_iter_7, up_acc_1)):
                        k0_11: pl.Scalar[pl.INDEX] = kb_5 * 512
                        post_chunk_0: pl.Tensor[[4, 512], pl.BFLOAT16] = pl.tensor.view(post_norm_tile_2, [4, 512], [0, k0_11])
                        wg_0: pl.Tensor[[512, 512], pl.BFLOAT16] = pl.tensor.view(w_gate_0, [512, 512], [k0_11, o0_3])
                        wu_0: pl.Tensor[[512, 512], pl.BFLOAT16] = pl.tensor.view(w_up_0, [512, 512], [k0_11, o0_3])
                        _t11: pl.Tensor[[4, 512], pl.BFLOAT16] = pl.tensor.matmul(post_chunk_0, wg_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                        gate_acc_4: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.add(gate_acc_iter_2, _t11)
                        _t12: pl.Tensor[[4, 512], pl.BFLOAT16] = pl.tensor.matmul(post_chunk_0, wu_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                        up_acc_4: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.add(up_acc_iter_2, _t12)
                        gate_acc_3, k0_10, up_acc_3 = pl.yield_(gate_acc_4, k0_11, up_acc_4)
                    _t13: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.neg(gate_acc_3)
                    _t14: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.exp(_t13)
                    _t15: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.add(_t14, 1.0)
                    sigmoid_0: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.recip(_t15)
                    _t16: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.mul(gate_acc_3, sigmoid_0)
                    mlp_chunk_0: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.mul(_t16, up_acc_3)
                    mlp_chunk_bf16_0: pl.Tensor[[4, 512], pl.BFLOAT16] = pl.tensor.cast(mlp_chunk_0, target_type=pl.BFLOAT16, mode=2)
                    for dob_0_out, (down_proj_tile_iter_4_outer_l0,) in pl.range(0, 7, 1, init_values=(down_proj_tile_iter_2,)):
                        down_proj_tile_iter_4_outer_l1_rv: pl.Tensor[[4, 7168], pl.FP32] = self.deepseek_v3_2_prefill_back_layer_incore_1(dob_0_out, down_proj_tile_1, down_proj_tile_iter_2, down_proj_tile_iter_4_outer_l0, mlp_chunk_bf16_0, o0_3, w_down_0)
                        down_proj_tile_iter_4_outer_l0_rv: pl.Tensor[[4, 7168], pl.FP32] = pl.yield_(down_proj_tile_iter_4_outer_l1_rv)
                    down_proj_tile_3, k0_8, kb_4, o0_2 = pl.yield_(down_proj_tile_iter_4_outer_l0_rv, k0_10, kb_5, o0_3)
                for ob_2_out, (o0_iter_4_outer_l0, out_iter_5_outer_l0) in pl.range(0, 7, 1, init_values=(o0_2, out_iter_3)):
                    ret: pl.Tuple([pl.Scalar[pl.INDEX], pl.Tensor[[16, 4096, 7168], pl.BFLOAT16]]) = self.deepseek_v3_2_prefill_back_layer_incore_2(b_0, down_proj_tile_3, o0_2, o0_iter_4_outer_l0, ob_2_out, out_0, out_iter_1, out_iter_3, out_iter_5_outer_l0, p0_0, resid1_tile_iter_1_outer_l0_rv)
                    o0_iter_4_outer_l1_rv: pl.Scalar[pl.INDEX] = ret[0]
                    out_iter_5_outer_l1_rv: pl.Tensor[[16, 4096, 7168], pl.BFLOAT16] = ret[1]
                    o0_iter_4_outer_l0_rv, out_iter_5_outer_l0_rv = pl.yield_(o0_iter_4_outer_l1_rv, out_iter_5_outer_l1_rv)
                out_4: pl.Tensor[[16, 4096, 7168], pl.BFLOAT16] = pl.yield_(out_iter_5_outer_l0_rv)
            out_2: pl.Tensor[[16, 4096, 7168], pl.BFLOAT16] = pl.yield_(out_4)
        return out_2