# DeepSeek v3.2 Decode Front SRAM Summary

## Source
- Report file: `deepseek_v3_2_decode_front_dump/report/memory_after_AllocateMemoryAddr.txt`
- Pass: `AllocateMemoryAddr`
- Backend: `910B_CCE`

## InCore Memory Snapshot
| InCore Function | Vec Used | Vec Limit | Usage | MemRefs |
|---|---:|---:|---:|---:|
| `deepseek_v3_2_decode_front_layer_incore_0_aiv` | 16 B | 192.0 KB | 0.0% | 1 |
| `deepseek_v3_2_decode_front_layer_incore_2_aiv` | 16 B | 192.0 KB | 0.0% | 1 |
| `deepseek_v3_2_decode_front_layer_incore_3_aiv` | 0 B | 192.0 KB | 0.0% | 4 |

## Notes
- 该统计反映的是当前Pass阶段的显式Vec空间占用；并不等价于最终硬件运行时所有临时缓冲峰值。
- 从现有报告看，Vec空间未构成瓶颈，后续优化优先级应放在算子融合与访存路径稳定性（尤其是codegen阻塞点）上。

