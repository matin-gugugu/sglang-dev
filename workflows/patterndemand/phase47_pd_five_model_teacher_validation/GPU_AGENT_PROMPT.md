# Phase47 GPU Agent任务

你是远程GPU执行Agent。先完整阅读本目录`README_CN.md`、`experiment.json`、`models.json`和所有脚本，再核验当前HEAD严格等于控制端给出的W47，并从W47创建唯一run分支。不要改变正式分支。

按README先在formal preflight之前获取五个固定revision。允许这一获取步骤联网，也允许使用操作者已有的HF授权下载gated Llama；禁止在聊天、命令、日志、结果或Git中显示token。模型必须放在独立持久化目录，不得覆盖其他用户缓存或受保护资产。

正式preflight和run必须设置`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，并让仓库`python/`成为`PYTHONPATH`第一项。选择一对空闲且不同的B200级GPU和一个ACTIVE IB HCA；五个模型顺序复用这对GPU。先preflight，再一条run命令，再verify。W47-fix4必须使用新的attempt4或更高编号外置目录，保留attempt1–3全部证据。

你可以在`AUTO`范围诊断端口占用、同类GPU/HCA、依赖和模型路径，失败重试必须使用新的raw/smoke目录并保留旧证据。不得换模型/revision/backend/page size/transport，不能改P/D内部TP/PP，不能关闭wave barrier，不能降低场景或重复数，不能删异常模型。两个page-budget smoke必须全部运行且各重复两次，不得只保留与teacher一致的形状。KV cache保持BF16语义，命令行固定使用TRTLLM MLA接受的`--kv-cache-dtype bf16`拼写，不得改回`bfloat16`或`auto`。任一科学精确门失败时不得把它包装成PASS。

PASS后只选择性添加Phase47正式结果目录，运行`verify_staging.py --phase phase47`，形成唯一父提交为W47的单一result commit R47并push run分支。回传W47、R47、run分支、环境/GPU/拓扑/HCA、五模型revision/backend/page、外置model/raw/smoke位置、每模型请求/chunk/精确匹配、README/summary/logs/DONE/manifest、允许和不允许得出的结论。禁止`git add .`。
