# Phase42 workflow：纯PD残差预测器训练与blind预测冻结

Phase42不运行SGLang、不使用GPU、不读取`data/`、raw、完整请求或blind target。它只能在不含受保护数据的干净Git worktree中读取Phase41已经提交的75个训练画像、19个开发验证画像和12个target-free blind画像。

候选选择只在75个训练画像内部做确定性五折OOF。选定配置后，用全部75个训练画像训练三个固定seed模型；19个开发验证画像只评估一次，不能反向改变配置。随后立即冻结12个blind画像的H0和H0+DNN residual预测。

推荐执行：

```bash
export CUDA_VISIBLE_DEVICES=-1
python3 workflows/patterndemand/phase42_pd_residual_training/preflight.py --expected-workflow-commit W42
python3 workflows/patterndemand/phase42_pd_residual_training/run.py --expected-workflow-commit W42
python3 workflows/patterndemand/phase42_pd_residual_training/verify.py
```

正式结果只允许选择性添加`experiment-results/phase42_pd_residual_training/`。checkpoint使用紧凑gzip JSON，不包含raw、完整请求或blind标签。R42必须以W42为唯一父提交。R42合入正式分支以前，禁止创建或运行Phase43。
