# 给本地控制端Agent的Phase44提示词

从W44创建独立run worktree，显式传入主仓库受保护raw绝对路径。先核验selection复现、历史embargo、六个raw SHA和所有pinned inputs，再运行CPU teacher与NumPy训练。

不得读取Phase43 label/per-profile metrics作为训练输入，不得改1200窗口、960/240划分、候选、alpha、fold、hard gate或在验证结果后重跑选择。不得使用GPU/网络，不得将完整请求、raw、权重缓存或PID加入Git。

成功后只能选择性添加`experiment-results/phase44_pd_expanded_protected_training/`。无论`model_accepted`真假都如实形成以W44为唯一父提交的R44；只有true才允许下一阶段冻结全新blind预测。
