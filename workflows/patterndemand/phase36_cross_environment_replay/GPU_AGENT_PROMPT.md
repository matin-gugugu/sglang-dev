# 给GPU环境Agent的Phase36提示词

请先完整阅读`workflows/patterndemand/PatternDemand跨环境GPU执行交接_Phase36_Phase37.md`和本目录`experiment.json`。从交接方给出的workflow commit `W`创建独立run分支，不要修改workflow或冻结输入。

运行：

```bash
python3 workflows/patterndemand/phase36_cross_environment_replay/run.py --expected-workflow-commit W
python3 workflows/patterndemand/phase36_cross_environment_replay/verify.py
```

Phase36不训练、不读取teacher/target，只需要一张GPU。若preflight阻塞，不得绕过；将证据写入正式目录的`BLOCKED.json`后只提交该目录。成功后只能选择性添加`experiment-results/phase36_cross_environment_replay/`，不得使用`git add .`。result commit必须以W为唯一父提交，push run分支并回传分支名、commit R、目录大小、文件数和manifest校验结果。
