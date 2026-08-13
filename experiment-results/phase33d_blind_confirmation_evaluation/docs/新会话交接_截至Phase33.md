# 新会话交接：截至Phase33

## 当前状态

Phase33已完成。TP和PP都在9个全新BurstGPT盲测窗口上正式通过；不再继续针对本确认集调参。

- TP最佳：`tp33_c10_shared_trunk_small_heads_policy_lr0.003_w64_5fold_3seed_alpha0.75`。
- PP最佳：`pp33_anchor_blend_1_phase32_incumbent_calls_shape`。
- 冻结预测SHA：`f634a6f0c82c0109132e752e4f52f266b9aa8904d947874ce98b32e9dd80f7d0`。
- Phase33C target-free归档：`284dcaef`；Phase33D盲测归档：`3222f972`。

## 必读目录

- `experiment-results/phase33a_fresh_data_contract/`
- `experiment-results/phase33b_expanded_development_dataset/`
- `experiment-results/phase33c_target_free_model_selection/`
- `experiment-results/phase33d_blind_confirmation_evaluation/`

## 关键边界

完整请求列表只用于离线Hfull teacher。最终输入仍是低维历史画像、模型结构、固定并行配置与策略。bytes近零误差来自允许输入下的结构锚点，不是DNN对任意bytes规律的泛化。新盲测只有BurstGPT；TP在旧重复集仍失败；PP MB16仍需作为重点回归切片。

下一阶段建议冻结首阶段接口，接入placement/topology连续通信代价曲线；扩到6个模型时沿用相同的数据隔离、训练和一次性盲测协议。
