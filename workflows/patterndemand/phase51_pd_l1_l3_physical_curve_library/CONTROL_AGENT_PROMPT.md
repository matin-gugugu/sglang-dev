# Phase51 控制端验收合同

只接受唯一父提交为W51、路径完全位于`experiment-results/phase51_pd_l1_l3_physical_curve_library/`的R51。先运行本目录`verify.py`，再运行`verify_result_commit.py --phase phase51 --workflow-commit W51 --result-commit R51`。

重点核验36个冻结measurement、18条曲线、396个knots、双方向/双replica保守取值、5/7/9次自适应重复、RDMA/dma-buf无fallback以及Git外raw manifest。不得把运行方差或L1/L2/L3非单调当成自动失败，但必须保留诊断状态。通过后才可ff-only合入正式分支；R51成为Phase52的父结果。Phase52只做冻结直方图与物理曲线的确定性卷积和communication-only placement验证，不重训。
