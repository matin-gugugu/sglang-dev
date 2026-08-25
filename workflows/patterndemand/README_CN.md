# PatternDemand跨环境实验workflow

这里的workflow不是GitHub Actions，而是给GPU或控制环境中的Agent执行的、可审计的实验合同。

每个workflow分为两层：

1. 不可改变的合同：研究语义、冻结输入、指标口径、最小重复次数、结果格式和Git边界。
2. 可自主判断的执行手册：Agent可以诊断环境、选择同类GPU对、有限重试、增加warmup或重复次数，但必须写入`decision_log.jsonl`，且不得改变实验问题。

统一执行顺序：

```text
workflow commit W
  -> 对应执行环境从W创建run分支
  -> preflight
  -> run（一条命令）
  -> verify
  -> 只添加允许的结果目录
  -> 单个result commit R
  -> push run分支
  -> 原环境验证R的父提交、路径和manifest
```

以下列出已冻结的跨环境主链workflow；Phase54–59为后续本地CPU精度探索，细节见各自目录：

- `phase36_cross_environment_replay`：一张GPU即可，不训练、不读teacher，复播Phase34冻结的六模型TP/PP直方图并演练commit回传。
- `phase37_pp_single_node_p2p_curve`：至少两张GPU，实测单机PP GPU P2P连续曲线；raw逐次样本保存在仓库外，只提交紧凑曲线与审计。
- `phase38_pp_physical_curve_cost_recompute`：Phase37结果验收并合入后，在CPU上将Phase34冻结PP直方图与Hfull target确定性代入物理P2P曲线；不加载checkpoint、不重训。本workflow只有在R37合入后才能提交形成W38，以保持固定W36/W37的ff-only结果链。
- `phase39_tp_pp_l1_l3_physical_placement_validation`：在测量前冻结L1/L2/L3 host/rack/rank placement，以24个分布式shard补全TP2/4/8与PP的物理曲线矩阵；随后在CPU上完成冻结直方图卷积、proxy对照及communication-only placement agreement/regret验证。raw仍在Git外，TP/PP size始终是输入而不是调度决策。
- `phase40_pure_pd_semantics_teacher`：纯P1→D1的语义与teacher基础闭环。固定Mooncake/RDMA、FCFS、chunk/cache/overlap口径，以45个代表请求核对真实sender-side KV chunk、模型结构字节公式、完整请求teacher和12-bin直方图；不训练、不做六模型扩展、物理曲线或调度器。
- `phase41_pd_full_window_dataset`：将Phase40语义扩展为最多64请求的有界fixed-draining wave，先以63/64/65/129边界和三个真实完整窗口做GPU sentinel；精确通过后才生成94个Qwen3开发画像的Hfull/H0/residual数据，并冻结12个不含完整请求和target的新盲测画像。仍不训练DNN。
- `phase42_pd_residual_training`：在不含raw的本地CPU隔离worktree中，仅用75个训练画像做五折候选选择，19个开发验证画像一次性报告表现，并在target打开前冻结12个blind画像的H0与H0+DNN residual预测。
- `phase43_pd_blind_evaluation`：R42正式冻结预测后，才在本地CPU控制端从六个受保护raw源重建12个blind完整窗口、生成Hfull标签并一次性评分；不重训、不加载checkpoint、不重算预测，完整请求仍不进入Git。
- `phase44_pd_expanded_protected_training`：避开Phase27–34及已打开Phase41/43窗口，冻结1200个互不重叠的BurstGPT开发画像，用CPU teacher扩展到960/240训练验证集；训练带有界残差、alpha收缩和分segment H0硬保护门的DNN，只有四项指标严格改善才允许新blind。
- `phase45_pd_fresh_blind_prediction_freeze`：在不生成Hfull的前提下冻结300个全新、历史隔离且分层的BurstGPT blind画像，并原样加载R44 checkpoint生成H0与H0+DNN预测；R45合入前禁止打开标签。
- `phase46_pd_fresh_blind_evaluation`：R45正式合入后才重建同一批300个窗口并核对冻结特征，随后一次性生成Hfull，按预注册四指标、三segment保护门、paired bootstrap和请求量分层报告blind结论；不训练或重算预测。
- `phase47_pd_five_model_teacher_validation`：在Qwen3-8B预测器通过fresh blind后，顺序复用同一对GPU，补齐DeepSeek-V2-Lite、Qwen3-30B-A3B、Llama-3.2-3B-Instruct、Qwen2.5-14B-Instruct和Mixtral-8x7B-Instruct的纯P1→D1 scheduler/逻辑KV teacher精确验证；固定DeepSeek TRTLLM MLA/page64及其余模型FlashInfer/page1，page>1时按SGLang真实整页占用扣prefill预算并使用双wave交叉smoke，不训练、不测物理时间。
- `phase48_pd_six_model_expanded_training`：把Phase41的Qwen开发画像按六种已验证模型结构确定性展开，在CPU上训练一个共享的H0保护残差模型；冻结模型差异为输入，不读取fresh blind标签。
- `phase49_pd_six_model_blind_prediction_freeze`：冻结300个fresh blind低维画像及六模型的H0/H0+DNN预测；不打开完整请求或Hfull，形成一次性盲测的预测先验。
- `phase50_pd_six_model_blind_evaluation`：R49合入后才重建完整请求和六模型Hfull，一次性验证整体、逐模型和逐segment的H0保护改善；不训练或重算预测。
- `phase51_pd_l1_l3_physical_curve_library`：不加载模型，按六模型真实KV描述符布局，直接调用SGLang生产Mooncake/RDMA batch-transfer测量L1/L2/L3；36个冻结shard汇总成18条模型相关物理曲线、396个knots，供后续Phase52确定性卷积与communication-only placement验证。
- `phase52_pd_physical_cost_placement_validation`：在R51合入后，用CPU将Phase49冻结的H0/H0+DNN、Phase50 Hfull与Phase51六模型L1/L2/L3物理曲线做确定性bin-mean卷积，报告物理cost误差、communication-only placement agreement/regret、双replica区间robust性和单调包络敏感性；不重训、不使用GPU。
- `phase53_tp_pp_pd_conclusion_freeze`：在R52合入后，本地CPU核验Phase34D至Phase52共19个正式结果commit与manifest，生成新的TP/PP/PD统一总导引、证据索引和结论边界；保留Phase43负结果，冻结Phase39/52的communication-only口径，不产生新预测、标签、物理测量或scheduler结果。
- `phase60_pd_multi_endpoint_composability`：冻结P1D1预测链和Phase51曲线，使用Qwen3-8B与DeepSeek-V2-Lite实测P1D2/P2D1两路同wave的L1/L2/L3 Mooncake/RDMA行为；以同批次solo锚点区分真实contention与环境漂移，只生成development可组合性证据，不拟合修正项或打开未来blind pair。
- `phase61_pd_contention_correction`：在本地CPU上只使用Phase60的120个development official point，以leave-one-payload-pair-out选择最简单达标的P1D2/P2D1 contention修正公式；冻结后才允许Phase62打开reserved payload和未见placement做GPU fresh-blind。
- `phase62_pd_contention_fresh_blind`：冻结R61全局max/min修正公式，在生产Mooncake RDMA上只测Phase60预留的20个reserved payload pair，并使用与Phase60 endpoint tuple完全不重合、每种拓扑至少一套host signature全新的placement；不训练、不调参，机械验证P1D2/P2D1两流修正是否达到整体10%、各配置拓扑15%的fresh-blind合同。
- `phase63_pd_contention_four_model_external`：冻结R61修正公式，对Qwen3-30B-A3B、Llama-3.2-3B、Qwen2.5-14B和Mixtral-8x7B四个held-out KV布局做P1D2/P2D1、L1–L3外部物理验证；48个三rank shard，与R62合并形成六模型证据。
- `phase64_pd_multiflow_graph_zero_shot`：冻结Phase51单链路曲线与R61系数，零训练实测P1D4、P4D1、P2D2 matching/all-to-all四种最多四流通信图；Qwen3-8B/DeepSeek-V2-Lite、L1–L3、双placement共48个shard，单shard最多两节点五进程。

完整交接见`PatternDemand跨环境GPU执行交接_Phase36_Phase37.md`。
