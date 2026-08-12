# Phase 28C：PP 冻结混合映射第二独立确认

状态：**PASS**。Phase 28B 的预测文件先以SHA-256冻结并提交Git，本阶段核验
hash和manifest之后，才由18个此前未用于Phase16/27的历史窗口生成Hfull teacher。没有
训练、早停、调参或改变 `MB1=H0，MB4/MB16=enhanced_bounded_residual` 映射。

## 18个第二确认画像的total结果

| 方法 | calls MAPE/WAPE | bytes MAPE/WAPE | TV | norm EMD | common cost MAPE/WAPE |
|---|---:|---:|---:|---:|---:|
| H0 | 62.05% / 18.85% | 2.70% / 1.25% | 0.2606 | 0.0458 | 10.97% / 6.47% |
| 冻结混合映射 | 24.82% / 9.42% | 3.08% / 1.64% | 0.1885 | 0.0296 | 5.25% / 3.35% |

## 分microbatch结果

- mb1：冻结方法 `h0`；calls MAPE 6.37%（H0 6.37%），TV 0.0801（H0 0.0801），cost MAPE 2.36%（H0 2.36%）。
- mb4：冻结方法 `enhanced_bounded_residual`；calls MAPE 13.59%（H0 34.45%），TV 0.1083（H0 0.1878），cost MAPE 5.13%（H0 8.96%）。
- mb16：冻结方法 `enhanced_bounded_residual`；calls MAPE 54.49%（H0 145.33%），TV 0.3771（H0 0.5139），cost MAPE 8.26%（H0 21.60%）。

对比图见 `figures/frozen_mapping_second_confirmation.png`。

Hfull标签来自完整窗口请求列表的GPU验证结构公式，完整请求列表只在内存中使用，未写入
结果目录或Git；保存的是324条归一化teacher phase rows。冻结映射的无偏结论只适用于
Qwen3-8B、PP2/4/8、fixed-draining和当前三种microbatch策略。common cost仍是5 μs +
100 GB/s统一参考曲线，不能当作PP P2P物理链路实测；也不能外推到跨模型PP或online
arrival-aware调度。
