# Phase65 控制端验收

R65必须只有W65一个父提交，只能修改`experiment-results/phase65_pd_graph_correction_development/`白名单文件。先运行`verify_result_commit.py --phase phase65`，再运行本目录`verify.py`。

验收重点：R64负结果完整保留；240行development数据；10-fold payload OOF与3-fold topology OOF；第一个达标候选；10%/15%门未降低；Phase66预留page与Phase64零重叠；无GPU/网络/新测量/Phase66 target。`PASS_TARGET_NOT_MET`也是有效执行结果，但不允许进入Phase66。
