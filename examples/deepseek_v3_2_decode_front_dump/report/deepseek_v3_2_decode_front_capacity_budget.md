# DeepSeek Front Capacity Budget

## Source
- File: `/data/liaoheng/pypto_workspace/pypto-lib/examples/deepseek_v3_2_decode_front.py`
- Kernel kind: `decode`

## Formula
- `stage1_est_bytes = BATCH_TILE*K_CHUNK*4 + BATCH_TILE*LORA_CHUNK*4 + BATCH_TILE*Q_OUT_CHUNK*4 + BATCH_TILE*KV_OUT_CHUNK*4`
- `stage2_est_bytes = 2*(1+2)*INDEX_TOPK*4 + KV_LORA_RANK*4 + QK_ROPE_HEAD_DIM*4 + V_HEAD_DIM*4`
- `peak_est_bytes = max(stage1_est_bytes, stage2_est_bytes)`

## Current Config
- `BATCH_TILE = 4`
- `K_CHUNK = 512`
- `LORA_CHUNK = 128`
- `Q_OUT_CHUNK = 512`
- `KV_OUT_CHUNK = 128`
- `LOCAL_PAD_WIDTH = 16384`
- `INDEX_TOPK = 2048`
- `KV_LORA_RANK = 512`
- `QK_ROPE_HEAD_DIM = 64`
- `V_HEAD_DIM = 128`
- `UB_SOFT_LIMIT_BYTES = 163840`

## Result
- `stage1_est_bytes = 151552 B`
- `stage2_est_bytes = 51968 B`
- `peak_est_bytes = 151552 B`
- `peak / UB_SOFT_LIMIT = 92.50%`
- Status: PASS

## Cube Usage (Estimated)
- `cube_tile_est_bytes = 536576 B`
- `cube_soft_limit_bytes = 1048576 B`
- `cube_usage_est = 51.17%`
