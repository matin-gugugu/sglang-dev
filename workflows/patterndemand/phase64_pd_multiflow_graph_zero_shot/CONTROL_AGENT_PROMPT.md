# Phase64 控制端验收合同

收到 R64 后，先用 `workflows/patterndemand/verify_result_commit.py --phase phase64 --workflow-commit W64 --result-commit R64` 验证唯一父提交、路径边界、禁止资产与 commit 内 manifest；再 checkout R64 运行本目录 `verify.py`。只有两层都通过，才允许 ff-only 合入正式分支并 push。

验收时单独检查：48 个 shard、240 official points、两个代表模型、四种图、L1/L2/L3、两个 placement；每个 measurement repeat 数只能是 5/7/9；raw 不在 Git；R61 系数未变；没有训练、推理、权重或下载；资源峰值为两节点/五进程/一个 shard。

科学 gate 失败不是 result commit 失败。只要执行与审计合同完整，R64 应保留真实 FAIL 结论；不得让 GPU Agent 针对 Phase64 标签调整公式后重跑同一 blind/development 集。
