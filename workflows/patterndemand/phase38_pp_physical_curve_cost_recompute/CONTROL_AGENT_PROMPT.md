# 给控制/CPU环境Agent的Phase38提示词

前置条件：Phase36和Phase37均已完成，R37已由控制端验证唯一父提交、路径、禁止资产和commit内manifest，并ff-only合入正式分支。Phase38 workflow随后单独提交，交接方给出该提交`W38`。不要从当前固定的`W36=72c81b53`直接运行Phase38。

先完整阅读本目录`README_CN.md`和`experiment.json`，从`W38`创建独立run分支。Phase38不需要GPU；不得加载checkpoint、运行模型推理、读取raw trace、下载模型权重或训练。

运行：

```bash
export CUDA_VISIBLE_DEVICES=""
python3 workflows/patterndemand/phase38_pp_physical_curve_cost_recompute/run.py \
  --expected-workflow-commit W38
python3 workflows/patterndemand/phase38_pp_physical_curve_cost_recompute/verify.py
```

preflight会重新验证R37、Phase37 manifest和物理曲线合同，并在结果内冻结R37、曲线SHA与所有Phase34/35输入SHA。preflight中的前置环境/输入合同不一致必须`BLOCKED`；计算后的行数、直方图不变性或有限值检查不通过则是`FAIL`，不得修改或筛选结果来变成PASS。不得换曲线、使用Phase35 proxy代替物理数据或降低校验条件。确认阻塞证据后可用下列命令生成紧凑回传目录，证据中不得包含密钥或大体积数据：

```bash
python3 workflows/patterndemand/record_blocked.py --phase phase38 \
  --reason '不可绕过的合同阻塞' --evidence-json '{"detail":"简明证据"}'
```

成功后只允许选择性添加：

```bash
git add -- experiment-results/phase38_pp_physical_curve_cost_recompute/
python3 workflows/patterndemand/verify_staging.py --phase phase38
```

禁止`git add .`。result commit必须以`W38`为唯一父提交；push run分支后回传`W38`、`R38`、run分支、执行环境、Phase37来源提交/状态/拓扑、结果目录、数据量、核心物理cost指标、README/summary/logs/DONE/manifest，以及可以和不可以得出的结论。
