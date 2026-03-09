# DeepSeek v3.2 Prefill Front SRAM Summary

## Source
- Report file: `deepseek_v3_2_prefill_front_dump/report/memory_after_AllocateMemoryAddr.txt`
- Pass: `AllocateMemoryAddr`
- Backend: `910B_CCE`

## InCore Memory Snapshot
| InCore Function | Vec Used | Vec Limit | Usage | MemRefs |
|---|---:|---:|---:|---:|
| `deepseek_v3_2_prefill_front_layer_incore_0_aiv` | 16 B | 192.0 KB | 0.0% | 1 |
| `deepseek_v3_2_prefill_front_layer_incore_1_aiv` | 0 B | 192.0 KB | 0.0% | 4 |

## Notes
- 该版本做了前端局部融合试验，`prefill_front` 的 mixed kernels 从 3 个变为 2 个（`incore_0/1`）。
- 该报告属于中间编译阶段统计；若要做最终峰值评估，需在后端 codegen 问题修复后补运行时统计。

