# 新会话完整交接（截至Phase32F）

## 当前结论

- 基础定义不变：低维历史画像、模型结构、固定执行策略和既定TP/PP配置作为输入；Hfull只作离线teacher；目标是fixed-draining拓扑无关消息直方图与同一连续通信代价曲线。
- TP最佳：`tp32_rescue_policy_phase_alpha1.0`，仍为H0+DNN residual。新确认重复工程calls/bytes/TV/EMD/cost WAPE=12.19%/2.63%/0.2045/0.0206/8.57%，累计48组绝对上限，裁定fail。
- PP最佳：`pp32_c18_pp_bytes_cost_protection_policy_lr0.003_w64_5fold_3seed_alpha0.75`，仍为H0+DNN residual。一次性新确认calls/bytes/TV/EMD/cost WAPE=4.51%/3.98%/0.1426/0.0183/3.48%，裁定conditional_pass。

## 数据与证据边界

- 原59个正常互斥画像不变；新增9个BurstGPT请求级互斥确认窗口，2,976个完整teacher请求。新增确认不含Mooncake。
- Phase32C是PP主证据和TP常规模型的一次性确认；Phase32D之后TP评测只能称重复工程证据。所有选模仍只用开发侧数据。

## 仓库

- 分支：`experiment/pattern-demand-v0.5.15-clean`
- node55：`/sgl-workspace/sglang-src`
- 本地：`/Users/liyafei06/Documents/Codex/2026-07-21/login-klingai-wlf2-ge151-node55-idchb2az2/work/sglang-phase2-curve`
- 最终目录：`experiment-results/phase32f_tp_pp_expanded_convergence_final`

## 必须保护

继续保护本地`data/`、远端Phase16 GPU目录、Phase19 formal-v1/v2/smoke与PID、Phase23历史PID/tmp、raw trace、缓存和所有PID；不得使用`git add .`。

## 下一步

不要继续在已打开的Phase32 target上调TP。若扩到6个模型，应先增加模型结构特征与开发teacher、统一重训TP/PP，再冻结新的互斥确认集。PP可作为当前incumbent。
