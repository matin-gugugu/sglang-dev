# Phase 31G：今晚TP/PP收敛最终汇总

## 最终裁定

- TP：**未收口（fail）**。最佳模型仍为Phase31C共享`H0+DNN residual`，固定集calls WAPE为14.33%、cost WAPE为8.99%；相对H0分别改善26.47%和28.64%，但仍超过有条件阈值12%/6%。Phase31E已将TP搜索补足至18组上限，新模型没有在验证集超过incumbent，因此停止搜索。
- PP：**有条件收口（conditional pass）**。最佳模型为按MB小头的`H0+DNN residual`；固定集calls WAPE 6.91%、bytes WAPE 4.05%、TV 0.1593、EMD 0.0199、cost WAPE 5.42%。calls/cost相对H0改善47.99%/42.53%。
- 整体第一阶段：**尚未完全收口**，原因只在TP；PP已达到今晚定义的有条件通过。

## 数据与模型

- 59个请求级互斥正常历史画像：39训练、10验证、10固定预测；三个已知模型均覆盖训练、验证和固定预测；
- 开发Hfull teacher使用21,058个完整窗口请求，固定评测使用2,786个请求；开发/固定target分别5,292/1,080条phase rows；
- TP覆盖TP2/4/8与latency/balanced/throughput；PP覆盖PP2/4/8与MB1/4/16；
- 完整请求只用于离线teacher，不是预测输入；最终输入仍是低维画像、模型结构、固定并行配置和策略。

## 每模型固定集指标

| 方向 | 模型 | calls WAPE | bytes WAPE | TV | EMD | cost WAPE |
|---|---|---:|---:|---:|---:|---:|
| TP | deepseek-v2-lite | 14.40% | 3.04% | 0.2191 | 0.0207 | 9.91% |
| TP | qwen3-8b | 14.09% | 2.89% | 0.2968 | 0.0305 | 7.48% |
| TP | qwen3-30b-a3b | 14.47% | 3.05% | 0.2214 | 0.0209 | 9.98% |
| PP | deepseek-v2-lite | 6.84% | 4.11% | 0.1670 | 0.0208 | 5.52% |
| PP | qwen3-8b | 7.05% | 4.07% | 0.1444 | 0.0184 | 5.22% |
| PP | qwen3-30b-a3b | 6.83% | 3.96% | 0.1665 | 0.0207 | 5.56% |

PP MB16 calls MAPE由104.16%降至36.02%，相对改善65.42%，达到单独保护条件。

## 可以得出的结论

在当前三个已知模型和正常历史流量范围内，PP的`H0+DNN residual`对fixed-draining消息直方图及统一参考通信代价具有稳定价值，并达到有条件收口；TP residual也在三个模型上方向一致地改善calls和cost，但绝对误差还不足以宣告收口。

## 不可以得出的结论

不能声称TP已经通过；不能声称对未见模型、极端流量或所有生产环境具备零样本泛化；Phase31F是同一固定集重复一致性评测，不是新盲测。

## 保存位置

- 本地：`/Users/liyafei06/Documents/Codex/2026-07-21/login-klingai-wlf2-ge151-node55-idchb2az2/work/sglang-phase2-curve/experiment-results/phase31g_tp_pp_convergence_final`；
- node55：`/sgl-workspace/sglang-src/experiment-results/phase31g_tp_pp_convergence_final`；
- 逐模型、逐policy及整体指标分别在`analysis/per_model_metrics.csv`、`analysis/per_policy_metrics.csv`和`analysis/headline_metrics.csv`。

## 下一步

不要继续在已打开的固定集上调TP。下一轮应先冻结新的请求级互斥确认集，再只用开发侧扩充正常窗口或改进低维顺序特征/形状residual；若仍保持今晚边界，则当前诚实结论就是“PP有条件收口、TP未收口”。
