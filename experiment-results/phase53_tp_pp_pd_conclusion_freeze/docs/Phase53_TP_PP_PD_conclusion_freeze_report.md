# Phase53：TP、PP、PD实验链与当前结论冻结报告

## 1. Phase53做了什么

Phase53没有训练、推理、teacher重算、GPU通信测量或scheduler仿真。它在workflow commit `446ebc0cf2cf7ebe757fea7b9a331580825defbf` 上核验Phase34D至Phase52共19个正式结果目录的manifest、result commit和状态，并把截至Phase52的三条实验链写成一个新的规范入口。

它解决的是“以后引用哪一个结果、能说到哪里”的问题，不产生新的科学样本。

## 2. 三条链的冻结状态

|链|预测器|teacher|曲线|代价|placement|边界|
|---|---|---|---|---|---|---|
|TP|Phase34D六模型H0+DNN residual fresh-blind PASS|scheduler-faithful Hfull；完整请求仅离线生成标签|Phase39 TP2/4/8 × L1/L2/L3，共9条物理曲线|Phase39 WAPE：L1 7.57%，L2 7.52%，L3 7.85%|Phase39 communication-only top1=100.00%，regret=0|固定TP/PP配置；target已打开；不含计算、显存、资源和重叠|
|PP|Phase34D六模型H0+DNN residual fresh-blind PASS|scheduler-faithful Hfull；完整请求仅离线生成标签|Phase39 PP × L1/L2/L3，共3条物理曲线|Phase39 WAPE：L1 4.41%，L2 3.99%，L3 4.22%|Phase39 communication-only top1=100.00%，regret=0|固定TP/PP配置；target已打开；不含计算、显存、资源和重叠|
|PD|Phase50六模型×300画像H0+DNN residual fresh-blind PASS，composite ratio=0.8961|Phase40/47 GPU精确验证的纯P1-D1 fixed-draining Hfull teacher|Phase51 18条六模型L1/L2/L3 Mooncake/RDMA物理曲线|Phase52 H0+DNN cost WAPE：L1 2.15%，L2 2.16%，L3 2.15%，三层均优于H0|Phase52 agreement 86.44%→87.22%；mean regret 0.0221%→0.0185%|纯P1-D1、fixed-draining；bin-mean卷积；不含在线到达、实例数或完整scheduler|

## 3. 冻结结论

|ID|冻结结论|正式证据|
|---|---|---|
|F01|最终预测输入是常态历史流量的低维画像、模型结构、固定执行策略和固定并行配置，不含完整请求列表。|Phase26-35与Phase40-50合同|
|F02|预测目标是在fixed-draining语义下的拓扑无关12-bin消息调用数和逻辑字节直方图。|Phase34D、Phase40/47、Phase50|
|F03|TP/PP size与纯PD的P1/D1均是预测器输入；当前placement模块只选冻结的L1/L2/L3，不选并行度。|Phase35、Phase39、Phase52|
|F04|Hfull是经过代表性GPU实验验证的scheduler-faithful离线teacher；完整请求只用于离线标签。|TP/PP teacher链与Phase40/41/47|
|F05|TP在六个已知模型的fresh-blind集合上保留H0+DNN residual并正式优于H0。|Phase34D|
|F06|PP在六个已知模型的fresh-blind集合上保留H0+DNN residual并正式优于H0。|Phase34D|
|F07|纯PD六模型teacher的请求级、聚合级和12-bin语义已与GPU sender-side事件精确对齐。|Phase40、Phase47|
|F08|纯PD在300画像×六模型fresh-blind集合上，H0+DNN residual的四项主直方图指标均严格优于H0。|Phase49/50|
|F09|TP2/4/8和PP的L1/L2/L3物理通信曲线已在冻结环境中补全，可用于Phase34直方图的确定性代价卷积。|Phase39|
|F10|纯PD六模型L1/L2/L3 Mooncake/RDMA物理曲线库已完成，共18条曲线和396个knots。|Phase51|
|F11|在固定并行配置与冻结候选placement内，TP/PP和PD均完成了communication-only cost与placement验证。|Phase39、Phase52|
|F12|Phase43的小样本负结果没有被删除；Phase44-50通过扩大互斥开发/盲测和H0保护门后才形成最终正结论。|Phase43-50|

## 4. 证据层级与替代关系

1. 新盲测预测结论：TP/PP以Phase34D为准；纯PD六模型以Phase50为准。
2. 物理曲线：TP/PP以Phase39为准；纯PD以Phase51为准。
3. communication-only cost/placement：TP/PP以Phase39为准；纯PD以Phase52为准。
4. Phase35的TP L2/L3与PP L1/L2/L3 proxy只保留为接口演进；不得覆盖Phase39物理结果。
5. Phase37/38是PP单机先导，Phase39给出最终冻结的TP/PP L1-L3矩阵。
6. Phase43是有效负结果；Phase46和Phase50是采用新开发/新blind协议后的后续正结果，不是删除或重算Phase43。

## 5. 禁止越界

- `N01` 不能宣称对未见第七模型或任意新模型泛化。
- `N02` 不能宣称覆盖所有流量分布、所有policy或任意线上工作负载。
- `N03` 12-bin卷积不能恢复bin内每条消息的精确物理代价。
- `N04` 不能把通信代价写成端到端请求延迟或吞吐收益。
- `N05` 当前placement没有处理计算时间和显存可行性。
- `N06` 当前placement没有处理资源空闲、排队、拥塞或通信计算重叠。
- `N07` fixed-draining结果不证明online arrival-aware调度。
- `N08` Phase39/52不是完整scheduler验证，也不是线上收益实验。
- `N09` 纯PD结果不包含P或D内部TP/PP，也不证明混合并行PD。
- `N10` 调度器尚不能选择TP/PP size、P/D实例数或扩缩容策略。

## 6. 正式来源提交

- Phase34D：`0c4058f0dcdc18f6d273f20914563b3ebbec2383`，状态 `PASS`，six-model TP/PP predictor confirmation。
- Phase35：`d7e74bea2b2f9de4f9d1cb169e25a7e487a85f7d`，状态 `PASS`，unified replay and proxy/physical cost interface。
- Phase36：`3b8682fcebf772fc73fc26de8c21de5d3369c62f`，状态 `PASS`，cross-environment frozen replay and result return。
- Phase37：`6e6c74d8433aa09bde6d8314993c97418630daf8`，状态 `PASS_WITH_LIMITED_TOPOLOGY`，first PP single-node physical P2P curve。
- Phase38：`ca6d52a291585b052737e8309f6da1a33fbb382b`，状态 `PASS`，frozen PP histogram physical-cost recomputation。
- Phase39：`e5697c03fffb9f9cebdc6beb2bc667d0aae5173a`，状态 `PASS_WITH_RUNTIME_AND_PLACEMENT_VARIANCE`，TP/PP L1-L3 physical curves and communication-only placement。
- Phase40：`0af31cda42b72042e1fe5a6173a440878c1a50e6`，状态 `PASS`，pure P1-D1 scheduler-faithful teacher foundation。
- Phase41：`914f7fb53c4ff076a5fe12f7c56624502673eb1f`，状态 `PASS`，full-window dataset and target-free pilot blind freeze。
- Phase42：`88dd1a8f5a4b9452e226118ade270aa3eb6fed7e`，状态 `PASS`，first Qwen3 residual pilot。
- Phase43：`4e627c8f72f2568111410d57f053eb82d61b1538`，状态 `PASS`，12-profile pilot blind negative evidence。
- Phase44：`61773b3d85f9f5c4cdce1ee92b4c287b92810f06`，状态 `PASS`，1200-profile protected development expansion。
- Phase45：`284f4b796b57bfee5002efb52937da26d0fe748f`，状态 `PASS`，300-profile Qwen3 fresh-blind predictions before targets。
- Phase46：`927a693d723a91a5a248ce332899164686e31601`，状态 `PASS`，Qwen3 protected residual confirmation。
- Phase47：`9de1816e912056a0a0b2b91d940079540ad6454a`，状态 `PASS`，five additional model teacher validation。
- Phase48：`f573507abd1b59fa08cdf09030d2f62048c7ee5c`，状态 `PASS`，six-model shared residual training。
- Phase49：`1b9227753f941cf9c790af69bf0acb7cf8bc3796`，状态 `PASS`，six-model 300-profile fresh-blind predictions before targets。
- Phase50：`a8ed946e3afa9d96c71a18fb1ea7aa155fdf3e57`，状态 `PASS`，six-model pure-PD protected residual confirmation。
- Phase51：`1f69c5e9b2f53af3fb96711875c828b2061bf156`，状态 `PASS_WITH_RUNTIME_AND_PLACEMENT_VARIANCE`，six-model PD L1-L3 Mooncake/RDMA curve library。
- Phase52：`400157f46b6e04eabbfa52097b7489b7fa89bd2d`，状态 `PASS`，PD physical cost and communication-only placement validation。

## 7. 后续治理

- 当前TP、PP、PD预测器停止在已打开target上继续调参。
- 当前物理曲线保留环境、primitive、布局、payload和placement限定。
- 新scheduler研究必须把计算、显存、资源、排队/拥塞、重叠和受约束L2/L3决策写入新合同。
- 新研究可以消费冻结直方图与曲线，但不能把scheduler收益倒写成预测器的新盲测结论。
