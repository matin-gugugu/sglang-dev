# Phase48 控制端执行合同

只在正式分支 exact W48、工作树除受保护 `data/` 外干净时执行。先运行单测和 preflight；不得使用 GPU、网络或模型权重，不得读取 Phase45/46 结果来调参，不得写入或修改六份 raw。

按 README 命令生成结果后运行 `verify.py`。结果只能选择性添加 `experiment-results/phase48_pd_six_model_expanded_training/`；禁止 `git add .`，禁止添加 `data/`、完整请求、JSONL、raw、缓存、PID、权重。结果 commit 必须以 W48 为唯一父提交，并由正式分支 ff-only 合入。

若 `model_accepted=false`，诚实保存结果但停止：不允许构造 Phase49。若为 true，先验收并合入 R48，之后另写 W49，冻结全新且 target-free 的六模型 blind 预测。
