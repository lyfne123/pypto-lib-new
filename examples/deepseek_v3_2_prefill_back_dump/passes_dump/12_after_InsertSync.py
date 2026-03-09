# pypto.program: DeepSeekV32PrefillBack
import pypto.language as pl

@pl.program
class DeepSeekV32PrefillBack:
    @pl.function(type=pl.FunctionType.InCore)
    def deepseek_v3_2_prefill_back_layer_incore_2(self, b_0: pl.Scalar[pl.INDEX], down_proj_tile_3: pl.Tensor[[4, 7168], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 114688, 0)], o0_2: pl.Scalar[pl.INDEX], o0_iter_4_outer_l0: pl.Scalar[pl.INDEX], ob_2_out: pl.Scalar[pl.INDEX], out_0: pl.Tensor[[16, 4096, 7168], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 939524096, 1)], out_iter_1: pl.Tensor[[16, 4096, 7168], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 939524096, 2)], out_iter_3: pl.Tensor[[16, 4096, 7168], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 939524096, 3)], out_iter_5_outer_l0: pl.Tensor[[16, 4096, 7168], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 939524096, 4)], p0_0: pl.Scalar[pl.INDEX], resid1_tile_iter_1_outer_l0_rv: pl.Tensor[[4, 7168], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 114688, 5)]) -> tuple[pl.Scalar[pl.INDEX], pl.Tensor[[16, 4096, 7168], pl.BFLOAT16]]:
        for ob_2_in, (o0_iter_4_outer_l1, out_iter_5_outer_l1) in pl.parallel(0, 8, 1, init_values=(o0_iter_4_outer_l0, out_iter_5_outer_l0)):
            o0_6: pl.Scalar[pl.INDEX] = (0 + (ob_2_out * 8 + ob_2_in) * 1) * 128
            _t18: pl.Tensor[[4, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 6)] = pl.tensor.view(down_proj_tile_3, [4, 128], [0, o0_6])
            _t19: pl.Tensor[[4, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 7)] = pl.tensor.view(resid1_tile_iter_1_outer_l0_rv, [4, 128], [0, o0_6])
            down_acc_0: pl.Tensor[[4, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 8)] = pl.tensor.add(_t18, _t19)
            _t20: pl.Tensor[[4, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 9)] = pl.tensor.cast(down_acc_0, target_type=pl.BFLOAT16, mode=2)
            out_7: pl.Tensor[[16, 4096, 7168], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 939524096, 10)] = pl.tensor.assemble(out_iter_5_outer_l1, _t20, [b_0, p0_0, o0_6])
            o0_iter_4_outer_l1_rv, out_iter_5_outer_l1_rv = pl.yield_(o0_6, out_7)
        return o0_iter_4_outer_l1_rv, out_iter_5_outer_l1_rv
    @pl.function(type=pl.FunctionType.Orchestration)
    def deepseek_v3_2_prefill_back_layer(self, hidden_states_0: pl.Tensor[[16, 4096, 7168], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 939524096, 0)], seq_lens_0: pl.Tensor[[16], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 1)], node_id_t_0: pl.Tensor[[1], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 4, 2)], combine_buf_0: pl.Tensor[[128, 16, 4096, 16384], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 274877906944, 3)], wo_0: pl.Tensor[[16384, 7168], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 234881024, 4)], post_rms_weight_0: pl.Tensor[[1, 7168], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 28672, 5)], w_gate_0: pl.Tensor[[7168, 18432], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 264241152, 6)], w_up_0: pl.Tensor[[7168, 18432], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 264241152, 7)], w_down_0: pl.Tensor[[18432, 7168], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 264241152, 8)], out_0: pl.Tensor[[16, 4096, 7168], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 939524096, 9)]) -> pl.Tensor[[16, 4096, 7168], pl.BFLOAT16]:
        node_id_0: pl.Scalar[pl.INT32] = pl.tensor.read(node_id_t_0, [0])
        for b_0, (out_iter_1,) in pl.parallel(0, 16, 1, init_values=(out_0,), chunk=4):
            seq_len_b_0: pl.Scalar[pl.INT32] = pl.tensor.read(seq_lens_0, [b_0])
            tok_blocks_0: pl.Scalar[pl.INDEX] = (pl.cast(seq_len_b_0, pl.INDEX) + 4 - 1) // 4
            for p0_idx_0, (out_iter_3,) in pl.range(0, tok_blocks_0, 1, init_values=(out_iter_1,)):
                p0_0: pl.Scalar[pl.INDEX] = p0_idx_0 * 4
                valid_tok_0: pl.Scalar[pl.INDEX] = min(4, pl.cast(seq_len_b_0, pl.INDEX) - p0_0)
                _t0: pl.Tensor[[4, 16384], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 131072, 10)] = pl.tensor.view(combine_buf_0, [4, 16384], [node_id_0, b_0, p0_0, 0])
                combined_tile_0: pl.Tensor[[4, 16384], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 262144, 11)] = pl.tensor.cast(_t0, target_type=pl.FP32, mode=2)
                resid1_tile_0: pl.Tensor[[4, 7168], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 114688, 12)] = pl.tensor.create([4, 7168], dtype=pl.FP32)
                for ob_0_out, (resid1_tile_iter_1_outer_l0,) in pl.range(0, 7, 1, init_values=(resid1_tile_0,)):
                    resid1_tile_iter_1_outer_l1_rv: pl.Tensor[[4, 7168], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 114688, 13)] = self.call_group(deepseek_v3_2_prefill_back_layer_incore_0_group, b_0, combined_tile_0, hidden_states_0, ob_0_out, p0_0, resid1_tile_0, resid1_tile_iter_1_outer_l0, wo_0)
                    resid1_tile_iter_1_outer_l0_rv: pl.Tensor[[4, 7168], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 114688, 14)] = pl.yield_(resid1_tile_iter_1_outer_l1_rv)
                sq_sum_0: pl.Tensor[[4, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 16, 15)] = pl.tensor.create([4, 1], dtype=pl.FP32)
                sq_sum_1: pl.Tensor[[4, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 16, 16)] = pl.tensor.mul(sq_sum_0, 0.0)
                for kb_1, (k0_iter_1, sq_sum_iter_2) in pl.range(0, 14, 1, init_values=(k0_0, sq_sum_1)):
                    k0_3: pl.Scalar[pl.INDEX] = kb_1 * 512
                    x_chunk_0: pl.Tensor[[4, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 17)] = pl.tensor.view(resid1_tile_iter_1_outer_l0_rv, [4, 512], [0, k0_3])
                    _t5: pl.Tensor[[4, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 18)] = pl.tensor.mul(x_chunk_0, x_chunk_0)
                    _t6: pl.Tensor[[4, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 16, 19)] = pl.tensor.row_sum(_t5)
                    sq_sum_4: pl.Tensor[[4, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 16, 20)] = pl.tensor.add(sq_sum_iter_2, _t6)
                    k0_2, sq_sum_3 = pl.yield_(k0_3, sq_sum_4)
                _t7: pl.Tensor[[4, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 16, 22)] = pl.tensor.mul(sq_sum_3, 0.000139509)
                _t8: pl.Tensor[[4, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 16, 23)] = pl.tensor.add(_t7, 1e-06)
                inv_rms_0: pl.Tensor[[4, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 16, 24)] = pl.tensor.rsqrt(_t8)
                post_norm_tile_0: pl.Tensor[[4, 7168], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 57344, 25)] = pl.tensor.create([4, 7168], dtype=pl.BFLOAT16)
                down_proj_tile_0: pl.Tensor[[4, 7168], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 114688, 26)] = pl.tensor.create([4, 7168], dtype=pl.FP32)
                down_proj_tile_1: pl.Tensor[[4, 7168], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 114688, 27)] = pl.tensor.mul(down_proj_tile_0, 0.0)
                for kb_2, (k0_iter_4, post_norm_tile_iter_1, x_chunk_iter_1) in pl.range(0, 14, 1, init_values=(k0_2, post_norm_tile_0, x_chunk_0)):
                    k0_6: pl.Scalar[pl.INDEX] = kb_2 * 512
                    x_chunk_3: pl.Tensor[[4, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 28)] = pl.tensor.view(resid1_tile_iter_1_outer_l0_rv, [4, 512], [0, k0_6])
                    gamma_0: pl.Tensor[[1, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 29)] = pl.tensor.view(post_rms_weight_0, [1, 512], [0, k0_6])
                    _t9: pl.Tensor[[4, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 30)] = pl.tensor.row_expand_mul(x_chunk_3, inv_rms_0)
                    normed_0: pl.Tensor[[4, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 31)] = pl.tensor.col_expand_mul(_t9, gamma_0)
                    _t10: pl.Tensor[[4, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 32)] = pl.tensor.cast(normed_0, target_type=pl.BFLOAT16, mode=2)
                    post_norm_tile_3: pl.Tensor[[4, 7168], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 57344, 33)] = pl.tensor.assemble(post_norm_tile_iter_1, _t10, [0, k0_6])
                    k0_5, post_norm_tile_2, x_chunk_2 = pl.yield_(k0_6, post_norm_tile_3, x_chunk_3)
                for ob_1, (down_proj_tile_iter_2, k0_iter_7, kb_iter_3, o0_iter_1) in pl.range(0, 36, 1, init_values=(down_proj_tile_1, k0_5, kb_2, o0_0)):
                    o0_3: pl.Scalar[pl.INDEX] = ob_1 * 512
                    gate_acc_0: pl.Tensor[[4, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 36)] = pl.tensor.create([4, 512], dtype=pl.FP32)
                    up_acc_0: pl.Tensor[[4, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 37)] = pl.tensor.create([4, 512], dtype=pl.FP32)
                    gate_acc_1: pl.Tensor[[4, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 38)] = pl.tensor.mul(gate_acc_0, 0.0)
                    up_acc_1: pl.Tensor[[4, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 39)] = pl.tensor.mul(up_acc_0, 0.0)
                    for kb_5, (gate_acc_iter_2, k0_iter_9, up_acc_iter_2) in pl.range(0, 14, 1, init_values=(gate_acc_1, k0_iter_7, up_acc_1)):
                        k0_11: pl.Scalar[pl.INDEX] = kb_5 * 512
                        post_chunk_0: pl.Tensor[[4, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 40)] = pl.tensor.view(post_norm_tile_2, [4, 512], [0, k0_11])
                        wg_0: pl.Tensor[[512, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 524288, 41)] = pl.tensor.view(w_gate_0, [512, 512], [k0_11, o0_3])
                        wu_0: pl.Tensor[[512, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 524288, 42)] = pl.tensor.view(w_up_0, [512, 512], [k0_11, o0_3])
                        _t11: pl.Tensor[[4, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 43)] = pl.tensor.matmul(post_chunk_0, wg_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                        gate_acc_4: pl.Tensor[[4, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 44)] = pl.tensor.add(gate_acc_iter_2, _t11)
                        _t12: pl.Tensor[[4, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 45)] = pl.tensor.matmul(post_chunk_0, wu_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                        up_acc_4: pl.Tensor[[4, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 46)] = pl.tensor.add(up_acc_iter_2, _t12)
                        gate_acc_3, k0_10, up_acc_3 = pl.yield_(gate_acc_4, k0_11, up_acc_4)
                    _t13: pl.Tensor[[4, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 49)] = pl.tensor.neg(gate_acc_3)
                    _t14: pl.Tensor[[4, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 50)] = pl.tensor.exp(_t13)
                    _t15: pl.Tensor[[4, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 51)] = pl.tensor.add(_t14, 1.0)
                    sigmoid_0: pl.Tensor[[4, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 52)] = pl.tensor.recip(_t15)
                    _t16: pl.Tensor[[4, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 53)] = pl.tensor.mul(gate_acc_3, sigmoid_0)
                    mlp_chunk_0: pl.Tensor[[4, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 54)] = pl.tensor.mul(_t16, up_acc_3)
                    mlp_chunk_bf16_0: pl.Tensor[[4, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 55)] = pl.tensor.cast(mlp_chunk_0, target_type=pl.BFLOAT16, mode=2)
                    for dob_0_out, (down_proj_tile_iter_4_outer_l0,) in pl.range(0, 7, 1, init_values=(down_proj_tile_iter_2,)):
                        down_proj_tile_iter_4_outer_l1_rv: pl.Tensor[[4, 7168], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 114688, 56)] = self.call_group(deepseek_v3_2_prefill_back_layer_incore_1_group, dob_0_out, down_proj_tile_1, down_proj_tile_iter_2, down_proj_tile_iter_4_outer_l0, mlp_chunk_bf16_0, o0_3, w_down_0)
                        down_proj_tile_iter_4_outer_l0_rv: pl.Tensor[[4, 7168], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 114688, 57)] = pl.yield_(down_proj_tile_iter_4_outer_l1_rv)
                    down_proj_tile_3, k0_8, kb_4, o0_2 = pl.yield_(down_proj_tile_iter_4_outer_l0_rv, k0_10, kb_5, o0_3)
                for ob_2_out, (o0_iter_4_outer_l0, out_iter_5_outer_l0) in pl.range(0, 7, 1, init_values=(o0_2, out_iter_3)):
                    ret: pl.Tuple([pl.Scalar[pl.INDEX], pl.Tensor[[16, 4096, 7168], pl.BFLOAT16]]) = self.deepseek_v3_2_prefill_back_layer_incore_2(b_0, down_proj_tile_3, o0_2, o0_iter_4_outer_l0, ob_2_out, out_0, out_iter_1, out_iter_3, out_iter_5_outer_l0, p0_0, resid1_tile_iter_1_outer_l0_rv)
                    o0_iter_4_outer_l1_rv: pl.Scalar[pl.INDEX] = ret[0]
                    out_iter_5_outer_l1_rv: pl.Tensor[[16, 4096, 7168], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 939524096, 59)] = ret[1]
                    o0_iter_4_outer_l0_rv, out_iter_5_outer_l0_rv = pl.yield_(o0_iter_4_outer_l1_rv, out_iter_5_outer_l1_rv)
                out_4: pl.Tensor[[16, 4096, 7168], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 939524096, 61)] = pl.yield_(out_iter_5_outer_l0_rv)
            out_2: pl.Tensor[[16, 4096, 7168], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 939524096, 62)] = pl.yield_(out_4)
        return out_2
    @pl.function(type=pl.FunctionType.InCore)
    def deepseek_v3_2_prefill_back_layer_incore_0_aic(self, b_0: pl.Scalar[pl.INDEX], combined_tile_0: pl.Tensor[[4, 16384], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 262144, 0)], hidden_states_0: pl.Tensor[[16, 4096, 7168], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 939524096, 1)], ob_0_out: pl.Scalar[pl.INDEX], p0_0: pl.Scalar[pl.INDEX], resid1_tile_0: pl.Tensor[[4, 7168], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 114688, 2)], resid1_tile_iter_1_outer_l0: pl.Tensor[[4, 7168], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 114688, 3)], wo_0: pl.Tensor[[16384, 7168], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 234881024, 4)]) -> pl.Tensor[[4, 7168], pl.FP32]:
        pl.comm.aic_initialize_pipe()
        for ob_0_in, (resid1_tile_iter_1_outer_l1,) in pl.parallel(0, 8, 1, init_values=(resid1_tile_iter_1_outer_l0,)):
            o0_0: pl.Scalar[pl.INDEX] = (0 + (ob_0_out * 8 + ob_0_in) * 1) * 128
            for kb_0 in pl.range(0, 32, 1):
                k0_0: pl.Scalar[pl.INDEX] = kb_0 * 512
                a_chunk_0__h0: pl.Tensor[[2, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 5)] = pl.comm.tpop_from_aiv(0)
                a_chunk_0__h1: pl.Tensor[[2, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 6)] = pl.comm.tpop_from_aiv(1)
                a_chunk_0__tmp: pl.Tensor[[4, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 7)] = pl.tensor.create(__list__(4, 512), dtype=pl.BFLOAT16)
                a_chunk_0__mid: pl.Tensor[[4, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 8)] = pl.tensor.assemble(a_chunk_0__tmp, a_chunk_0__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                a_chunk_0: pl.Tensor[[4, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 9)] = pl.tensor.assemble(a_chunk_0__mid, a_chunk_0__h1, __list__(2, 0))
                pl.comm.tfree_to_aiv(1)
                w_chunk_0__h0: pl.Tensor[[256, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 10)] = pl.comm.tpop_from_aiv(0)
                w_chunk_0__h1: pl.Tensor[[256, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 11)] = pl.comm.tpop_from_aiv(1)
                w_chunk_0__tmp: pl.Tensor[[512, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 131072, 12)] = pl.tensor.create(__list__(512, 128), dtype=pl.BFLOAT16)
                w_chunk_0__mid: pl.Tensor[[512, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 131072, 13)] = pl.tensor.assemble(w_chunk_0__tmp, w_chunk_0__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                w_chunk_0: pl.Tensor[[512, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 131072, 14)] = pl.tensor.assemble(w_chunk_0__mid, w_chunk_0__h1, __list__(256, 0))
                pl.comm.tfree_to_aiv(1)
                _t2: pl.Tensor[[4, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 15)] = pl.tensor.matmul(a_chunk_0, w_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                __half0__: pl.Tensor[[2, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 16)] = pl.tensor.view(_t2, __list__(2, 128), __list__(0, 0))
                __half1__: pl.Tensor[[2, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 17)] = pl.tensor.view(_t2, __list__(2, 128), __list__(2, 0))
                pl.comm.tpush_to_aiv(__half0__, 0)
                pl.comm.tpush_to_aiv(__half1__, 1)
        return resid1_tile_iter_1_outer_l1_rv
    @pl.function(type=pl.FunctionType.InCore)
    def deepseek_v3_2_prefill_back_layer_incore_0_aiv(self, b_0: pl.Scalar[pl.INDEX], combined_tile_0: pl.Tensor[[4, 16384], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 262144, 0)], hidden_states_0: pl.Tensor[[16, 4096, 7168], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 939524096, 1)], ob_0_out: pl.Scalar[pl.INDEX], p0_0: pl.Scalar[pl.INDEX], resid1_tile_0: pl.Tensor[[4, 7168], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 114688, 2)], resid1_tile_iter_1_outer_l0: pl.Tensor[[4, 7168], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 114688, 3)], wo_0: pl.Tensor[[16384, 7168], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 234881024, 4)], AIV_IDX: pl.Scalar[pl.INDEX]) -> pl.Tensor[[4, 7168], pl.FP32]:
        pl.comm.aiv_initialize_pipe()
        for ob_0_in, (resid1_tile_iter_1_outer_l1,) in pl.parallel(0, 8, 1, init_values=(resid1_tile_iter_1_outer_l0,)):
            o0_0: pl.Scalar[pl.INDEX] = (0 + (ob_0_out * 8 + ob_0_in) * 1) * 128
            o_acc_0: pl.Tensor[[2, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 5)] = pl.tensor.create([2, 128], dtype=pl.FP32)
            o_acc_1: pl.Tensor[[4, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 7)] = pl.tensor.mul(o_acc_0, 0.0)
            for kb_0, (o_acc_iter_2,) in pl.range(0, 32, 1, init_values=(o_acc_1,)):
                k0_0: pl.Scalar[pl.INDEX] = kb_0 * 512
                _t1: pl.Tensor[[2, 512], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 8)] = pl.tensor.view(combined_tile_0, [2, 512], [0 + AIV_IDX * 2, k0_0])
                a_chunk_0: pl.Tensor[[2, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 10)] = pl.tensor.cast(_t1, target_type=pl.BFLOAT16, mode=2)
                pl.comm.tpush_to_aic(a_chunk_0, AIV_IDX)
                w_chunk_0: pl.Tensor[[256, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 12)] = pl.tensor.view(wo_0, [256, 128], [k0_0 + AIV_IDX * 256, o0_0])
                pl.comm.tpush_to_aic(w_chunk_0, AIV_IDX)
                _t2: pl.Tensor[[2, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 14)] = pl.comm.tpop_from_aic(AIV_IDX)
                o_acc_4: pl.Tensor[[2, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 16)] = pl.tensor.add(o_acc_iter_2, _t2)
                pl.comm.tfree_to_aic(AIV_IDX)
                o_acc_3: pl.Tensor[[4, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 18)] = pl.yield_(o_acc_4)
            _t3: pl.Tensor[[2, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 19)] = pl.tensor.view(hidden_states_0, [2, 128], [b_0 + AIV_IDX * 2, p0_0, o0_0])
            resid_0: pl.Tensor[[2, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 21)] = pl.tensor.cast(_t3, target_type=pl.FP32, mode=2)
            _t4: pl.Tensor[[2, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 23)] = pl.tensor.add(o_acc_3, resid_0)
            resid1_tile_3: pl.Tensor[[4, 7168], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 114688, 25)] = pl.tensor.assemble(resid1_tile_iter_1_outer_l1, _t4, [0 + AIV_IDX * 2, o0_0])
            resid1_tile_iter_1_outer_l1_rv: pl.Tensor[[4, 7168], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 114688, 26)] = pl.yield_(resid1_tile_3)
        return resid1_tile_iter_1_outer_l1_rv
    @pl.function_group(aic="deepseek_v3_2_prefill_back_layer_incore_0_aic", aiv="deepseek_v3_2_prefill_back_layer_incore_0_aiv", aiv_runtime_params=["AIV_IDX"])
    class deepseek_v3_2_prefill_back_layer_incore_0_group:
        """Parameter passing:
          call_group(deepseek_v3_2_prefill_back_layer_incore_0_group, b_0, combined_tile_0, hidden_states_0, ob_0_out, p0_0, resid1_tile_0, resid1_tile_iter_1_outer_l0, wo_0)
            → deepseek_v3_2_prefill_back_layer_incore_0_aic(b_0, combined_tile_0, hidden_states_0, ob_0_out, p0_0, resid1_tile_0, resid1_tile_iter_1_outer_l0, wo_0)
            → deepseek_v3_2_prefill_back_layer_incore_0_aiv(b_0, combined_tile_0, hidden_states_0, ob_0_out, p0_0, resid1_tile_0, resid1_tile_iter_1_outer_l0, wo_0, AIV_IDX=<runtime>)
        """
        pass

    @pl.function(type=pl.FunctionType.InCore)
    def deepseek_v3_2_prefill_back_layer_incore_1_aic(self, dob_0_out: pl.Scalar[pl.INDEX], down_proj_tile_1: pl.Tensor[[4, 7168], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 114688, 0)], down_proj_tile_iter_2: pl.Tensor[[4, 7168], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 114688, 1)], down_proj_tile_iter_4_outer_l0: pl.Tensor[[4, 7168], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 114688, 2)], mlp_chunk_bf16_0: pl.Tensor[[4, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 3)], o0_3: pl.Scalar[pl.INDEX], w_down_0: pl.Tensor[[18432, 7168], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 264241152, 4)]) -> pl.Tensor[[4, 7168], pl.FP32]:
        pl.comm.aic_initialize_pipe()
        for dob_0_in, (down_proj_tile_iter_4_outer_l1,) in pl.parallel(0, 8, 1, init_values=(down_proj_tile_iter_4_outer_l0,)):
            d0_0: pl.Scalar[pl.INDEX] = (0 + (dob_0_out * 8 + dob_0_in) * 1) * 128
            w_down_chunk_0__h0: pl.Tensor[[256, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 5)] = pl.comm.tpop_from_aiv(0)
            w_down_chunk_0__h1: pl.Tensor[[256, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 6)] = pl.comm.tpop_from_aiv(1)
            w_down_chunk_0__tmp: pl.Tensor[[512, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 131072, 7)] = pl.tensor.create(__list__(512, 128), dtype=pl.BFLOAT16)
            w_down_chunk_0__mid: pl.Tensor[[512, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 131072, 8)] = pl.tensor.assemble(w_down_chunk_0__tmp, w_down_chunk_0__h0, __list__(0, 0))
            pl.comm.tfree_to_aiv(0)
            w_down_chunk_0: pl.Tensor[[512, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 131072, 9)] = pl.tensor.assemble(w_down_chunk_0__mid, w_down_chunk_0__h1, __list__(256, 0))
            pl.comm.tfree_to_aiv(1)
            _t17: pl.Tensor[[4, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 10)] = pl.tensor.matmul(mlp_chunk_bf16_0, w_down_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
            __half0__: pl.Tensor[[2, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 11)] = pl.tensor.view(_t17, __list__(2, 128), __list__(0, 0))
            __half1__: pl.Tensor[[2, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 12)] = pl.tensor.view(_t17, __list__(2, 128), __list__(2, 0))
            pl.comm.tpush_to_aiv(__half0__, 0)
            pl.comm.tpush_to_aiv(__half1__, 1)
        return down_proj_tile_iter_4_outer_l1_rv
    @pl.function(type=pl.FunctionType.InCore)
    def deepseek_v3_2_prefill_back_layer_incore_1_aiv(self, dob_0_out: pl.Scalar[pl.INDEX], down_proj_tile_1: pl.Tensor[[4, 7168], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 114688, 0)], down_proj_tile_iter_2: pl.Tensor[[4, 7168], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 114688, 1)], down_proj_tile_iter_4_outer_l0: pl.Tensor[[4, 7168], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 114688, 2)], mlp_chunk_bf16_0: pl.Tensor[[4, 512], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 3)], o0_3: pl.Scalar[pl.INDEX], w_down_0: pl.Tensor[[18432, 7168], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 264241152, 4)], AIV_IDX: pl.Scalar[pl.INDEX]) -> pl.Tensor[[4, 7168], pl.FP32]:
        pl.comm.aiv_initialize_pipe()
        for dob_0_in, (down_proj_tile_iter_4_outer_l1,) in pl.parallel(0, 8, 1, init_values=(down_proj_tile_iter_4_outer_l0,)):
            d0_0: pl.Scalar[pl.INDEX] = (0 + (dob_0_out * 8 + dob_0_in) * 1) * 128
            down_prev_0: pl.Tensor[[2, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 5)] = pl.tensor.view(down_proj_tile_iter_4_outer_l1, [2, 128], [0, d0_0])
            w_down_chunk_0: pl.Tensor[[256, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 6)] = pl.tensor.view(w_down_0, [256, 128], [o0_3 + AIV_IDX * 256, d0_0])
            pl.comm.tpush_to_aic(w_down_chunk_0, AIV_IDX)
            _t17: pl.Tensor[[2, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 512, 8)] = pl.comm.tpop_from_aic(AIV_IDX)
            down_next_0: pl.Tensor[[2, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 1024, 11)] = pl.tensor.add(down_prev_0, _t17)
            pl.comm.tfree_to_aic(AIV_IDX)
            down_proj_tile_6: pl.Tensor[[4, 7168], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 114688, 13)] = pl.tensor.assemble(down_proj_tile_iter_4_outer_l1, down_next_0, [0 + AIV_IDX * 2, d0_0])
            down_proj_tile_iter_4_outer_l1_rv: pl.Tensor[[4, 7168], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 114688, 14)] = pl.yield_(down_proj_tile_6)
        return down_proj_tile_iter_4_outer_l1_rv
    @pl.function_group(aic="deepseek_v3_2_prefill_back_layer_incore_1_aic", aiv="deepseek_v3_2_prefill_back_layer_incore_1_aiv", aiv_runtime_params=["AIV_IDX"])
    class deepseek_v3_2_prefill_back_layer_incore_1_group:
        """Parameter passing:
          call_group(deepseek_v3_2_prefill_back_layer_incore_1_group, dob_0_out, down_proj_tile_1, down_proj_tile_iter_2, down_proj_tile_iter_4_outer_l0, mlp_chunk_bf16_0, o0_3, w_down_0)
            → deepseek_v3_2_prefill_back_layer_incore_1_aic(dob_0_out, down_proj_tile_1, down_proj_tile_iter_2, down_proj_tile_iter_4_outer_l0, mlp_chunk_bf16_0, o0_3, w_down_0)
            → deepseek_v3_2_prefill_back_layer_incore_1_aiv(dob_0_out, down_proj_tile_1, down_proj_tile_iter_2, down_proj_tile_iter_4_outer_l0, mlp_chunk_bf16_0, o0_3, w_down_0, AIV_IDX=<runtime>)
        """
        pass
