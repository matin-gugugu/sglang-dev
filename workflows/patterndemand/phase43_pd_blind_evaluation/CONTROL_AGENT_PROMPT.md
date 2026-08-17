# 给本地控制端Agent的Phase43提示词

确认正式HEAD已经是R42，从随后提交的精确W43创建独立run分支。完整阅读本目录`experiment.json`和`README_CN.md`。六个raw源只能作为Git外只读输入，先做bytes/SHA核验。

本阶段只允许：复现12个已冻结window ID；验证重建的低维feature/H0与Phase41逐字段一致；用固定teacher生成12行Hfull标签；将标签与R42冻结预测连接并评分。不得加载checkpoint、重新推理、训练、调参、换窗口、修改metric或把完整请求写入结果目录。

成功后只能选择性添加`experiment-results/phase43_pd_blind_evaluation/`，禁止`git add .`。R43必须以W43为唯一父提交。
