# Phase34 六模型扩展最终报告

Phase34 已完成，六模型 TP 和 PP 的 `H0 + 非零DNN residual` 均正式通过。

## 模型和数据

- 原模型：`deepseek-v2-lite`、`qwen3-8b`、`qwen3-30b-a3b`。
- 新模型：小型 dense `llama-3.2-3b-instruct`、深层大 hidden dense `qwen2.5-14b-instruct`、top-2 MoE `mixtral-8x7b-instruct-v0.1`。
- 开发集：94 个画像，75 train、19 validation，35,524 个完整 teacher 请求。
- 六模型样本：TP 10,152 条、PP 10,152 条；profile-grouped 五折。
- 新盲测：12 个全新且请求级互斥的正常 BurstGPT 窗口，3,803 个完整 teacher 请求。
- Phase34C 先归档预测、checkpoint 和 SHA，Phase34D 才打开 target；打开后没有调参。

## 最佳候选和新盲测

- TP：`tp34_c17_lowdim_cost_protected_gate_model_policy_lr0.001_w32_5fold_3seed_alpha1.0`。calls MAPE 9.12%、calls WAPE 7.81%、bytes WAPE约0、TV 0.1903、EMD 0.0201、cost MAPE 5.13%、cost WAPE 4.35%，正式通过。
- PP：`pp34_c04_pp_split_retrain_policy_lr0.003_w64_5fold_3seed_alpha1.0`。calls MAPE 14.88%、calls WAPE 4.53%、bytes WAPE约0、TV 0.1287、EMD 0.0171、cost MAPE 6.03%、cost WAPE 3.29%，正式通过。
- TP 相对 H0：calls WAPE 改善 41.41%，cost WAPE 改善 44.68%。
- PP 相对 H0：calls WAPE 改善 54.80%，cost WAPE 改善 54.55%。
- 六个模型 calls 无明显退化；PP MB16 calls MAPE 相对 H0 改善 69.82%，官方保护条件通过。

## Phase33 同窗比较

只比较同一 12 个新窗口上的原三个模型：TP calls/cost WAPE 相对 Phase33 均改善 7.42%；PP 均改善 29.67%。Phase33 原 9 个已打开窗口仅是重复工程证据。

## 边界

不能宣称未见第七模型泛化、跨所有流量分布、所有 policy 逐项过线或重新完成 GPU 实测。新盲测只覆盖 BurstGPT；bytes 近零来自六模型重新核验的低维均值结构锚点，不是 DNN 自由拟合。

冻结预测 SHA：`faffe08800e6336fa9272b765ca1965d4aee806a6162255f7bd9a50d5d5b5bda`。

Phase34C 提交：`6bdc0208e44e2cd8a51905560502e2ffe6c336f5`。Phase34D 首次归档提交：`0c4058f0dcdc18f6d273f20914563b3ebbec2383`。
