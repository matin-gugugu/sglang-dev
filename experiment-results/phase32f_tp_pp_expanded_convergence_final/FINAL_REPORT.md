# Phase 32A-F：TP/PP扩容收敛实验最终报告

## 最终裁定

- **TP：未收口（fail）**。最优仍是`H0+DNN residual`，累计搜索达到绝对上限48组。救援后的新确认重复工程结果为calls MAPE/WAPE=15.36%/12.19%、bytes MAPE/WAPE=2.73%/2.63%、TV=0.2045、EMD=0.0206、cost MAPE/WAPE=10.25%/8.57%。calls WAPE仅比12%有条件线高0.19个百分点，但cost WAPE仍比6%线高2.57个百分点，不能判为有条件通过。
- **PP：有条件收口（conditional pass）**。最优模型是`pp32_c18_pp_bytes_cost_protection_policy_lr0.003_w64_5fold_3seed_alpha0.75`。一次性新确认上calls/TV/EMD/cost与各模型保护均通过；bytes WAPE=3.98%高于正式3%线，因此不是正式通过。
- **第一阶段整体：部分收口**。PP已经满足今晚定义的有条件收口，TP方向一致改善但到达绝对搜索上限后仍未达到阈值。

## TP主结果（救援后重复工程证据）

| 方法 | calls MAPE/WAPE | bytes MAPE/WAPE | TV | EMD | cost MAPE/WAPE |
|---|---:|---:|---:|---:|---:|
| H0 | 21.60%/16.41% | 2.73%/2.63% | 0.2141 | 0.0214 | 14.29%/11.38% |
| H0+DNN residual | 15.36%/12.19% | 2.73%/2.63% | 0.2045 | 0.0206 | 10.25%/8.57% |

calls/cost WAPE相对H0改善25.73%/24.70%。需要强调：Phase32D选模完全只用开发侧分组OOF，但Phase32C已经先打开确认target，所以该数值是重复工程证据，不是新盲测。Phase32B在target开放前的一次性TP结果为calls WAPE 12.58%、cost WAPE 8.82%，同样裁定fail。

## PP主结果（一次性新确认）

| 方法 | calls MAPE/WAPE | bytes MAPE/WAPE | TV | EMD | cost MAPE/WAPE |
|---|---:|---:|---:|---:|---:|
| H0 | 36.52%/6.98% | 2.73%/2.63% | 0.2166 | 0.0295 | 15.66%/5.69% |
| H0+DNN residual | 17.50%/4.51% | 4.25%/3.98% | 0.1426 | 0.0183 | 7.38%/3.48% |

calls/cost WAPE相对H0改善35.30%/38.96%。PP MB16 calls MAPE相对H0改善54.18%，且bytes/TV/cost没有同时恶化超过10%，满足MB16保护。

## 每模型结果

| 方向 | 模型 | calls WAPE | bytes WAPE | TV | EMD | cost WAPE |
|---|---|---:|---:|---:|---:|---:|
| TP | deepseek-v2-lite | 12.04% | 2.63% | 0.1978 | 0.0186 | 9.10% |
| TP | qwen3-30b-a3b | 12.24% | 2.63% | 0.1975 | 0.0186 | 9.25% |
| TP | qwen3-8b | 12.23% | 2.63% | 0.2181 | 0.0246 | 7.54% |
| PP | deepseek-v2-lite | 4.54% | 3.95% | 0.1631 | 0.0199 | 3.66% |
| PP | qwen3-30b-a3b | 4.54% | 3.95% | 0.1628 | 0.0199 | 3.66% |
| PP | qwen3-8b | 4.46% | 4.01% | 0.1018 | 0.0151 | 3.16% |

## 数据、输入与隔离

- 既有59个请求级互斥正常画像保持不变：39训练、10验证、10原固定预测；新增9个BurstGPT正常确认窗口，与Phase27/28/30/31所有角色保持300秒请求区间隔离，且彼此互斥；原10个固定窗口没有更换。
- 开发teacher使用21,058个完整请求；原固定集2,786个请求；新增确认集2,976个请求。新增teacher为972条phase labels，完整请求列表没有保存或提交。
- 三个已知模型全部覆盖：deepseek-v2-lite、qwen3-8b、qwen3-30b-a3b。TP覆盖TP2/4/8与latency/balanced/throughput；PP覆盖PP2/4/8与MB1/4/16。
- 完整请求只用于离线Hfull teacher。最终预测输入仍是低维历史画像、模型结构、固定TP/PP配置与固定策略；输出仍是fixed-draining拓扑无关消息直方图，再代入同一连续通信代价曲线。
- 训练、alpha、gate、checkpoint与候选选择只使用训练/验证/5折profile分组OOF；固定target不进入上述任何环节。

## 搜索与停止原因

- TP：Phase31累计18组；Phase32B新增24组到常规上限42；Phase32D新增global/policy/model/phase/model×policy/policy×phase六组开发OOF gate，到绝对上限48后停止。
- PP：Phase31累计12组；Phase32B新增18组到常规上限30，覆盖bytes/cost保护loss与MB独立gate。一次性新确认达到有条件通过，按停止规则不再使用6组救援额度。
- 初筛每组1个seed；每方向前三组进行3-seed、5折确认。没有无边界搜索，也没有改变固定预测集或阈值。

## 可以得出的结论

在当前三个已知模型、正常历史流量、既定TP/PP配置与fixed-draining策略范围内，PP的H0+DNN residual已经给出可自洽的有条件收口证据；TP residual在全部三个模型上均改善calls和cost，但绝对cost误差仍阻止收口。结构H0仍是必要基线，DNN没有被取消。

## 不可以得出的结论

不能声称TP已经通过；不能把TP救援复评包装成新盲测；不能声称PP正式通过；不能外推到未见模型、极端流量或生产全域。新增确认只有BurstGPT，因为累计300秒隔离后Mooncake没有剩余完整窗口。

## 保存位置

- node55：`/sgl-workspace/sglang-src/experiment-results/phase32f_tp_pp_expanded_convergence_final`
- 本地：`/Users/liyafei06/Documents/Codex/2026-07-21/login-klingai-wlf2-ge151-node55-idchb2az2/work/sglang-phase2-curve/experiment-results/phase32f_tp_pp_expanded_convergence_final`
- 整体、逐模型、逐policy、逐并行规模、搜索清单分别见`analysis/`；checkpoint与冻结预测的实际路径和SHA见`checkpoints/`、`predictions/`。

## 下一步

今晚搜索已经按绝对上限停止。若后续继续TP，必须先冻结新的请求级互斥确认集，再从开发侧增加正常窗口或改进专门针对总量/代价的target-free residual；不能继续用已打开的Phase32确认target挑模型。PP保持当前incumbent，新增到6个模型时统一重训并重新确认。
