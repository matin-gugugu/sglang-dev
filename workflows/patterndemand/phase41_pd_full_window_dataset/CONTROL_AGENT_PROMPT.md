# Phase41控制端Agent提示词

你是PatternDemand控制端Agent。本任务只构建并传递Phase41 Git外bundle，不使用GPU、不生成正式结果commit。

1. 从用户指定的正式workflow commit `W41`开始，确认分支和HEAD；跟踪文件必须干净，只允许既有受保护`data/`未跟踪。
2. 完整阅读本目录`experiment.json`、`feature_contract.json`、`README_CN.md`和`GPU_AGENT_PROMPT.md`。
3. 使用受保护Phase15 raw目录只读执行`prepare_bundle.py`。不得移动、改写、重命名或删除raw；不得触碰旧GPU raw、PID、模型权重或cache。
4. bundle目录必须是Git外全新目录。记录`bundle_manifest.json`中的workflow commit、文件大小和SHA-256。
5. 通过当前环境获准的受控文件传输方式把整个bundle目录传给GPU Agent；传输后在GPU端重新核对大小和SHA。bundle不得通过Git、不得发到公开对象存储。
6. 不得打开或生成盲测Hfull目标。bundle中只能有94个开发窗口的完整请求；12个盲测窗口只能有低维画像。
7. 向GPU Agent只需提供：`W41`、bundle绝对路径、本地Qwen3-8B绝对路径、GPU pair、IB device、Git外raw目录、Git外preflight audit路径。其余语义由W41文档决定。

必须报告：W41、raw六文件hash PASS、94/35,524开发数据重建、12个盲测target-free、bundle路径/bytes/SHA、传输后核验状态。不要创建run分支，不要替GPU Agent运行Phase41。
