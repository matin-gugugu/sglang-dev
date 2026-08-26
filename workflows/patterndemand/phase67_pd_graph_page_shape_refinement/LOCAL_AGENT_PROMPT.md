# Phase67本地执行Agent提示词

以正式分支W67创建独立干净run分支；完整阅读本目录全部合同文件。只执行README_CN.md中的preflight、run和verify。不得使用GPU/网络，不得生成新物理数据，不得访问任何Phase68 measurement/target，不得修改候选、阈值、fold或输出。

verify通过后，严格按`commit_allowlist.txt`逐文件选择性添加结果；禁止`git add .`。结果commit必须只有W67一个父提交，再push run分支并报告W67、R67、状态、四种validation指标、选中公式、结果目录、manifest和结论边界。
