# 新会话完整交接（截至Phase31G）

## 当前结论

- 研究基础定义不变：低维历史画像、模型结构、固定执行策略及既定TP/PP配置作为输入；Hfull只作离线teacher；预测fixed-draining拓扑无关消息直方图，再代入连续通信代价曲线。
- 数据合同：59个请求级互斥正常画像，39训练、10验证、10固定预测；三个已知模型均覆盖三种角色。
- TP最佳：Phase31C `tp_c03_full_shared_lr0.003_3seed_alpha0.75`，固定calls/bytes/TV/EMD/cost为14.33%/2.97%/0.2458/0.0240/8.99%。TP搜索已到18组上限，裁定fail。
- PP最佳：Phase31C `pp_c05_full_policy_heads_lr0.001_3seed_alpha1.0`，固定calls/bytes/TV/EMD/cost为6.91%/4.05%/0.1593/0.0199/5.42%，裁定conditional_pass。

## 仓库与保存

- 分支：`experiment/pattern-demand-v0.5.15-clean`；
- node55仓库：`/sgl-workspace/sglang-src`；
- 本地仓库：`/Users/liyafei06/Documents/Codex/2026-07-21/login-klingai-wlf2-ge151-node55-idchb2az2/work/sglang-phase2-curve`；
- 最终总报告：`experiment-results/phase31g_tp_pp_convergence_final`；
- Phase31A-F提交：phase31a=217ad83bcc83, phase31b=6352f2a8b8c8, phase31c=eb9b8b7373ec, phase31d=1d182b0457f3, phase31e=0e0dbd90846a, phase31f=fde778a8bcc4。

## 必须继续保护

- 本地`data/`；
- 远端`experiment-results/phase16_profiledemand_gpu/`；
- 远端Phase19 formal-v1/v2/smoke与PID；
- 远端Phase23历史PID、server PID及tmp；
- 不使用`git add .`，不提交raw trace、完整请求列表、缓存或PID。

## 下一步

不要再用已打开的10个固定窗口调TP。若继续TP，应先target-blind冻结新的请求级互斥确认集，再扩充开发侧正常窗口或实现低维顺序/形状residual，并在新确认集上只评一次。PP已条件收口，不应继续搜索；后续可随模型从3个扩展到6个时统一重训。

## 结论边界

可以说PP在当前三个已知模型和正常流量范围内有用且条件收口，TP residual方向一致改善但未收口。不能说TP通过，也不能声称未见模型、极端流量或生产全域零样本泛化；Phase31F不是新盲测。
