# pypto.program: DeepSeekV32DecodeBack
import pypto.language as pl

@pl.program
class DeepSeekV32DecodeBack:
    @pl.function
    def deepseek_v3_2_decode_back_layer(self, hidden_states: pl.Tensor[[16, 7168], pl.BFLOAT16], node_id_t: pl.Tensor[[1], pl.INT32], combine_buf: pl.Tensor[[128, 16, 16384], pl.BFLOAT16], wo: pl.Tensor[[16384, 7168], pl.BFLOAT16], post_rms_weight: pl.Tensor[[1, 7168], pl.FP32], w_gate: pl.Tensor[[7168, 18432], pl.BFLOAT16], w_up: pl.Tensor[[7168, 18432], pl.BFLOAT16], w_down: pl.Tensor[[18432, 7168], pl.BFLOAT16], out: pl.Tensor[[16, 7168], pl.BFLOAT16]) -> pl.Tensor[[16, 7168], pl.BFLOAT16]:
        node_id: pl.Scalar[pl.INT32] = pl.tensor.read(node_id_t, [0])
        combined: pl.Tensor[[16, 16384], pl.FP32] = pl.tensor.create([16, 16384], dtype=pl.FP32)
        for b in pl.parallel(0, 16, 1, chunk=4):
            row: pl.Tensor[[1, 16384], pl.FP32] = pl.tensor.cast(pl.tensor.view(combine_buf, [1, 16384], [node_id, b, 0]), target_type=pl.FP32, mode=2)
            combined: pl.Tensor[[16, 16384], pl.FP32] = pl.tensor.assemble(combined, row, [b, 0])
        with pl.auto_incore():
            for b0 in pl.range(0, 16, 4):
                resid1_tile: pl.Tensor[[4, 7168], pl.FP32] = pl.tensor.create([4, 7168], dtype=pl.FP32)
                for ob in pl.parallel(0, 56, 1, chunk=8):
                    o0: pl.Scalar[pl.INDEX] = ob * 128
                    o_acc: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.create([4, 128], dtype=pl.FP32)
                    o_acc: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.mul(o_acc, 0.0)
                    for kb in pl.range(0, 32, 1):
                        k0: pl.Scalar[pl.INDEX] = kb * 512
                        a_chunk: pl.Tensor[[4, 512], pl.BFLOAT16] = pl.tensor.cast(pl.tensor.view(combined, [4, 512], [b0, k0]), target_type=pl.BFLOAT16, mode=2)
                        w_chunk: pl.Tensor[[512, 128], pl.BFLOAT16] = pl.tensor.view(wo, [512, 128], [k0, o0])
                        o_acc: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.add(o_acc, pl.tensor.matmul(a_chunk, w_chunk, a_trans=False, b_trans=False, c_matrix_nz=False))
                    resid: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.cast(pl.tensor.view(hidden_states, [4, 128], [b0, o0]), target_type=pl.FP32, mode=2)
                    resid1_tile: pl.Tensor[[4, 7168], pl.FP32] = pl.tensor.assemble(resid1_tile, pl.tensor.add(o_acc, resid), [0, o0])
                sq_sum: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.create([4, 1], dtype=pl.FP32)
                sq_sum: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.mul(sq_sum, 0.0)
                for kb in pl.range(0, 14, 1):
                    k0: pl.Scalar[pl.INDEX] = kb * 512
                    x_chunk: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.view(resid1_tile, [4, 512], [0, k0])
                    sq_sum: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.add(sq_sum, pl.tensor.row_sum(pl.tensor.mul(x_chunk, x_chunk)))
                inv_rms: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.rsqrt(pl.tensor.add(pl.tensor.mul(sq_sum, 0.000139509), 1e-06))
                post_norm_tile: pl.Tensor[[4, 7168], pl.BFLOAT16] = pl.tensor.create([4, 7168], dtype=pl.BFLOAT16)
                down_proj_tile: pl.Tensor[[4, 7168], pl.FP32] = pl.tensor.create([4, 7168], dtype=pl.FP32)
                down_proj_tile: pl.Tensor[[4, 7168], pl.FP32] = pl.tensor.mul(down_proj_tile, 0.0)
                for kb in pl.range(0, 14, 1):
                    k0: pl.Scalar[pl.INDEX] = kb * 512
                    x_chunk: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.view(resid1_tile, [4, 512], [0, k0])
                    gamma: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.view(post_rms_weight, [1, 512], [0, k0])
                    normed: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(x_chunk, inv_rms), gamma)
                    post_norm_tile: pl.Tensor[[4, 7168], pl.BFLOAT16] = pl.tensor.assemble(post_norm_tile, pl.tensor.cast(normed, target_type=pl.BFLOAT16, mode=2), [0, k0])
                for ob in pl.range(0, 36, 1):
                    o0: pl.Scalar[pl.INDEX] = ob * 512
                    gate_acc: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.create([4, 512], dtype=pl.FP32)
                    up_acc: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.create([4, 512], dtype=pl.FP32)
                    gate_acc: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.mul(gate_acc, 0.0)
                    up_acc: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.mul(up_acc, 0.0)
                    for kb in pl.range(0, 14, 1):
                        k0: pl.Scalar[pl.INDEX] = kb * 512
                        post_chunk: pl.Tensor[[4, 512], pl.BFLOAT16] = pl.tensor.view(post_norm_tile, [4, 512], [0, k0])
                        wg: pl.Tensor[[512, 512], pl.BFLOAT16] = pl.tensor.view(w_gate, [512, 512], [k0, o0])
                        wu: pl.Tensor[[512, 512], pl.BFLOAT16] = pl.tensor.view(w_up, [512, 512], [k0, o0])
                        gate_acc: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.add(gate_acc, pl.tensor.matmul(post_chunk, wg, a_trans=False, b_trans=False, c_matrix_nz=False))
                        up_acc: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.add(up_acc, pl.tensor.matmul(post_chunk, wu, a_trans=False, b_trans=False, c_matrix_nz=False))
                    sigmoid: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.recip(pl.tensor.add(pl.tensor.exp(pl.tensor.neg(gate_acc)), 1.0))
                    mlp_chunk: pl.Tensor[[4, 512], pl.FP32] = pl.tensor.mul(pl.tensor.mul(gate_acc, sigmoid), up_acc)
                    mlp_chunk_bf16: pl.Tensor[[4, 512], pl.BFLOAT16] = pl.tensor.cast(mlp_chunk, target_type=pl.BFLOAT16, mode=2)
                    for dob in pl.parallel(0, 56, 1, chunk=8):
                        d0: pl.Scalar[pl.INDEX] = dob * 128
                        down_prev: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.view(down_proj_tile, [4, 128], [0, d0])
                        w_down_chunk: pl.Tensor[[512, 128], pl.BFLOAT16] = pl.tensor.view(w_down, [512, 128], [o0, d0])
                        down_next: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.add(down_prev, pl.tensor.matmul(mlp_chunk_bf16, w_down_chunk, a_trans=False, b_trans=False, c_matrix_nz=False))
                        down_proj_tile: pl.Tensor[[4, 7168], pl.FP32] = pl.tensor.assemble(down_proj_tile, down_next, [0, d0])
                for ob in pl.parallel(0, 56, 1, chunk=8):
                    o0: pl.Scalar[pl.INDEX] = ob * 128
                    down_acc: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.add(pl.tensor.view(down_proj_tile, [4, 128], [0, o0]), pl.tensor.view(resid1_tile, [4, 128], [0, o0]))
                    out: pl.Tensor[[16, 7168], pl.BFLOAT16] = pl.tensor.assemble(out, pl.tensor.cast(down_acc, target_type=pl.BFLOAT16, mode=2), [b0, o0])
        return out