# Phase58 控制端交接

远端 Agent 从 W58 精确 HEAD 运行，不要登录 node55，不要申请 GPU，不要下载模型。运行前确认旧 W57-fix 进程已经停止，且 Phase58 结果目录不存在；不要让不同 workflow 并行写同一目录。

运行后先执行 `verify.py`，再报告 summary、round_trace、OOF bin bias、model/segment validation、实际耗时、run 分支、结果目录、DONE 和 manifest。若目标未达成，这是合法 development negative result；不得用 validation 或 Phase50 blind 再调参。

结果 commit 必须以 W58 为唯一父提交，控制端验收后再决定是否 ff-only 合入正式分支。禁止 `git add .`，禁止 raw、JSONL、完整请求、权重、cache、PID 和 `.pt/.pth/.safetensors`。
