# PatternDemand 新电脑完整交接（截至 Phase73）

> 本文用于在新电脑上让新的控制端 Agent 直接接手，不依赖旧聊天记忆。正式证据截止到 Phase73 结果提交 `a13e4ba83dc683edd3493d6d7aff03b207daa688`。导引/交接文档本身可能位于该提交之后的文档提交中。

## 1. 新电脑第一步

```bash
git clone --branch experiment/pattern-demand-v0.5.15-clean --single-branch \
  git@github.com:matin-gugugu/sglang-dev.git sglang-patterndemand
cd sglang-patterndemand
git status --short
git rev-parse HEAD
git log --oneline -5
```

必须确认：

- 当前分支为 `experiment/pattern-demand-v0.5.15-clean`；
- Phase73 结果提交 `a13e4ba8` 是当前 HEAD 的祖先；
- 新 clone 工作树干净；
- 不要根据旧机器绝对路径寻找资产，所有正式入口都应使用仓库相对路径。

随后完整阅读：

1. `docs/patterndemand/PatternDemand_experiment_guide_through_Phase73.md`
2. `experiment-results/phase72_conclusion_freeze_through_phase71/audit/claim_scope.json`
3. `experiment-results/phase73_direct_gbdt_baseline/README.md`
4. `workflows/patterndemand/README_CN.md`

## 2. 研究目标与口径

输入是常态历史流量的低维画像、模型结构、固定执行策略和已确定的 TP/PP/PD 配置；输出是 fixed-draining 语义下拓扑无关的 12-bin calls/bytes 消息直方图。物理曲线把直方图转换为 communication-only 通信代价。

- 并行配置是预测器输入；当前 placement 只在冻结 L1/L2/L3 候选中选择。
- 完整请求只用于离线 Hfull teacher 标签，不进入最终预测器。
- TP、PP、PD 主方法都是 `H0 + DNN residual`。
- H0 是结构先验；Direct-GBDT 是不使用 H0/伪请求/teacher 的独立 baseline。
- online arrival-aware、完整 scheduler 和 P/D 内部 TP/PP 都不在当前结论范围。

## 3. 当前正式状态

- 正式分支：`experiment/pattern-demand-v0.5.15-clean`
- Phase73 结果截止 HEAD：`a13e4ba83dc683edd3493d6d7aff03b207daa688`
- TP/PP：六模型 fresh-blind、L1–L3 物理曲线、communication-only placement 已完成。
- 纯 PD：六模型 teacher、六模型 H0+DNN、P1D1 L1–L3 曲线、两流/四流物理修正和固定 wave cost/placement 已完成。
- Phase59 严格 PD 直方图绝对门仍未达到：development calls/bytes histogram WAPE 18.93%/17.29%。
- Phase73 Direct-GBDT 没有优于 H0：calls/bytes histogram WAPE 30.79%/39.55%。
- 当前冻结范围内没有必须重跑的 GPU 实验。

## 4. 正式 Git 资产与 Git 外资产

### `sglang-dev` 正式仓库保存

- workflow、合同、校验脚本；
- 精简正式结果、预测、checkpoint、summary、README、logs、DONE、manifest；
- teacher/物理曲线的审计摘要与可复核结论；
- Phase72 总结和 Phase73 独立 baseline。

### 独立数据仓库保存

仓库：`git@github.com:matin-gugugu/patterndemand-data.git`

计划保存：

- `raw/phase15_traces/`：三份 BurstGPT 与三份 Mooncake 冻结原始请求轨迹；
- `archive/unmerged-results/phase54_*`、`phase55_*`、`phase56_*`：未纳入正式证据的本地探索结果，仅作历史备份；
- `archive/local-drafts/`：旧电脑上未提交的研究草稿和候选图；
- `archive/legacy-handoffs/`：旧会话交接与阶段报告；
- 每类资产的 SHA-256、来源、正式性说明和恢复方法。

该仓库与正式结果仓库用途不同。数据/归档仓库中的 Phase54–56 不能因为被备份就自动升级为正式实验结果。

### 仍留在 GPU 环境、不会随换电脑自动迁移

正式 Git 文件记录了若干 NVIDIA A51 外置 raw 位置，例如：

- `/lustre/fsw/coreai_comparch_infbench/huanhuanc/phase39_ext/run2/phase39_raw`
- `/lustre/fsw/coreai_comparch_infbench/huanhuanc/phase40_raw_fix4`
- `/lustre/fsw/coreai_comparch_infbench/huanhuanc/phase41_raw_attempt1`
- `/lustre/fsw/coreai_comparch_infbench/huanhuanc/phase51_ext/run1/raw`
- `/lustre/fsw/coreai_comparch_infbench/huanhuanc/phase60_ext/run1/raw`
- `/lustre/fsw/coreai_comparch_infbench/huanhuanc/phase62_ext/run1/raw`
- `/lustre/fsw/coreai_comparch_infbench/huanhuanc/phase63_ext/run1/raw`
- `/lustre/fsw/coreai_comparch_infbench/huanhuanc/phase64_pd_multiflow_graph_zero_shot/ext2/run1/raw`
- `/lustre/fsw/coreai_comparch_infbench/huanhuanc/phase66_ext/run1/raw`
- `/lustre/fsw/coreai_comparch_infbench/huanhuanc/phase68_ext/run1/raw`
- `/lustre/fsw/coreai_comparch_infbench/huanhuanc/phase70_ext/run1/raw`

还有 Phase47 五模型的 `patterndemand_runs/.../ext/raw_attempt4/`。这些 raw、模型权重和缓存没有进入 Git；正式结果中的 summary/manifest 已足够复核当前论文结论，但若要逐次重审物理样本，仍需保留 A51 存储访问权限或另行归档。

## 5. 不能丢、但不能混进正式分支的旧电脑资产

换机前已识别：

| 资产 | 状态 | 处理 |
|---|---|---|
| `data/phase15_traces/raw/`，约 400 MB | 六源哈希全部 PASS；两个文件超过普通 GitHub 单文件限制 | 用 Git LFS 存入独立私有数据仓库 |
| `experiment-results/phase54_*` 至 `phase56_*` | 本地未跟踪，Phase72 明确排除 | 归档到数据仓库的 `archive/unmerged-results/`，不加入正式结果树 |
| `docs/patterndemand/figures0820/`、`figures08201618/` | 候选图版本 | 归档到 `archive/local-drafts/` |
| 本地修改的 `kaiti_patternDemand实验.md` | 未提交研究草稿 | 保存副本，不覆盖正式仓库版本 |
| 本地修改的 `generate_patterndemand_figures.py` | 未提交制图实验 | 保存副本并标注非正式 |
| 旧 `outputs/` 下阶段报告/交接文档 | 仓库外历史资料 | 整体归档到 `archive/legacy-handoffs/` |

禁止使用 `git add .`。正式仓库和数据归档仓库都应逐路径选择性添加。

## 6. 受保护资产规则

- 不删除或覆盖旧机器 `data/`、远程 Phase16、Phase19 formal-v1/v2/smoke/PID、Phase23 保护目录；
- raw trace、Phase37/39/40/41/47/51/60–70 外置逐次样本、模型权重、缓存和 PID 不进入正式结果仓库；
- 新数据仓库只做备份和恢复，不改变实验正式性；
- 结果合入继续使用唯一父提交校验、禁止资产检查、commit 内 manifest 和 ff-only；
- target-open 结果不能写成 fresh-blind。

## 7. 新电脑恢复数据

安装 Git LFS 后：

```bash
git lfs install
git clone git@github.com:matin-gugugu/patterndemand-data.git
cd patterndemand-data
git lfs pull
python3 scripts/verify_assets.py
```

恢复到正式仓库时不要复制进 Git index。需要运行依赖 raw 的 workflow 时，通过参数或受保护的 `data/` 目录引用，并先核验 manifest。

## 8. 给新 Agent 的启动提示词

```text
请接手我的 SGLang PatternDemand 实验。不要让我重新复述历史，也不要立即运行实验。

先完整阅读：
1. docs/patterndemand/PatternDemand_experiment_guide_through_Phase73.md
2. docs/patterndemand/PatternDemand_new_computer_handoff_through_Phase73.md
3. experiment-results/phase72_conclusion_freeze_through_phase71/audit/claim_scope.json
4. experiment-results/phase73_direct_gbdt_baseline/README.md
5. workflows/patterndemand/README_CN.md

正式分支是 experiment/pattern-demand-v0.5.15-clean。先只读核验本地 HEAD、GitHub tracking 和工作树；确认 Phase73 结果提交 a13e4ba83dc683edd3493d6d7aff03b207daa688 是当前 HEAD 的祖先。不要使用 GPU，不要创建 run 分支，不要修改或添加 data/raw、模型权重、缓存和 PID。

当前结论：TP/PP/纯PD主链、L1-L3曲线及当前PD两流/四流 communication-only 集成已冻结；Phase59严格PD直方图绝对门未达到；Phase73 Direct-GBDT是target-open独立baseline且未优于H0；当前范围内没有必须重跑的GPU实验。

读完后先用自己的话报告研究目标、Hfull/H0/H0+DNN/Direct-GBDT区别、TP/PP/PD状态、可宣称与不可宣称结论、Git外资产位置，以及建议下一步。未经我确认不要扩大研究范围。
```

## 9. 新电脑验收清单

- [ ] 正式仓库 clone 成功，Phase73 result 是 HEAD 祖先；
- [ ] 两份新导引可读；
- [ ] `patterndemand-data` 完成 `git lfs pull`；
- [ ] 六个 raw 的 bytes 与 SHA-256 全部 PASS；
- [ ] Phase54–56 明确标为 archive/non-formal；
- [ ] 新 Agent 能复述 fixed-draining、12-bin、Hfull/H0/H0+DNN/Direct-GBDT 和当前 scheduler 边界；
- [ ] 没有把 A51 raw、模型权重或缓存误认为已在正式 Git 中。
