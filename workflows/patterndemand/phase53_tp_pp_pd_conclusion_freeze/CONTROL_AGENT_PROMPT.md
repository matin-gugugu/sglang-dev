# Phase53控制端执行提示词

从控制端指定的W53创建独立本地run分支和隔离worktree。确认HEAD严格等于W53且工作树干净后，依次执行`preflight.py`、`run.py`和`verify.py`。

本阶段不需要GPU或外网，不得登录GPU环境，不得加载模型/checkpoint，不得读取`data/`、raw trace或完整请求。不得修改Phase34D至Phase52任何既有结果。

提交前只按`commit_allowlist.txt`逐项暂存Phase53结果目录，运行`verify_staging.py --phase phase53`。result commit必须只有W53一个父提交；回传W53、R53、run分支、结果目录、manifest和verify证据。控制端再用`verify_result_commit.py --phase phase53`验收并ff-only合入。
