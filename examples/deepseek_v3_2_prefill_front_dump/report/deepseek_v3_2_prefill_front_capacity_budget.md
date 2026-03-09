# DeepSeek Front Capacity Budget

## Source
- File: `/data/liaoheng/pypto_workspace/pypto-lib/examples/deepseek_v3_2_prefill_front.py`
- Kernel kind: `prefill`

## Formula
- `stage1_est_bytes = TOK_TILE*K_CHUNK*4 + TOK_TILE*LORA_CHUNK*4 + TOK_TILE*Q_OUT_CHUNK*4 + TOK_TILE*KV_OUT_CHUNK*4`
- `stage2_est_bytes = 2*(1+2)*INDEX_TOPK*4 + KV_LORA_RANK*4 + QK_ROPE_HEAD_DIM*4 + V_HEAD_DIM*4`
- `peak_est_bytes = max(stage1_est_bytes, stage2_est_bytes)`

## Current Config
- `TOK_TILE = 4`
- `K_CHUNK = 512`
- `LORA_CHUNK = 128`
- `Q_OUT_CHUNK = 256`
- `KV_OUT_CHUNK = 128`
- `LOCAL_PAD_WIDTH = 16384`
- `INDEX_TOPK = 2048`
- `KV_LORA_RANK = 512`
- `QK_ROPE_HEAD_DIM = 64`
- `V_HEAD_DIM = 128`
- `UB_SOFT_LIMIT_BYTES = 163840`

## Result
- `stage1_est_bytes = 147456 B`
- `stage2_est_bytes = 51968 B`
- `peak_est_bytes = 147456 B`
- `peak / UB_SOFT_LIMIT = 90.00%`
- Status: PASS

## Cube Usage (Estimated)
- `cube_tile_est_bytes = 270336 B`
- `cube_soft_limit_bytes = 1048576 B`
- `cube_usage_est = 25.78%`
