# 给本地控制端Agent的Phase42提示词

从控制端指定的W42创建独立本地run分支和不含`data/`的干净worktree。完整阅读本目录`experiment.json`与`README_CN.md`。设置`CUDA_VISIBLE_DEVICES=-1`，在离线CPU环境依次运行preflight、run和verify。

不得把主仓库的受保护`data/`挂入执行worktree，不得读取Phase41外置bundle、任何完整请求、raw trace或blind target，不得修改候选网格、fold、seed、metric或根据19个validation结果重新训练。出现负结果也必须如实冻结12个blind预测。

成功后只能选择性添加`experiment-results/phase42_pd_residual_training/`，禁止`git add .`。result commit R42必须以W42为唯一父提交。控制端验收并ff-only合入R42后，才允许编写W43。
