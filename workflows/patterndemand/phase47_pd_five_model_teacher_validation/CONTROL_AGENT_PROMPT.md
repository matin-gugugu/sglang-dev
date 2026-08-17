# Phase47 控制端验收

收到R47后，先获取run分支但不要合并。运行：

```bash
python3 workflows/patterndemand/verify_result_commit.py \
  --phase phase47 --workflow-commit W47 --result-commit R47
git worktree add /TMP/phase47-verify R47
python3 /TMP/phase47-verify/workflows/patterndemand/phase47_pd_five_model_teacher_validation/verify.py
```

确认R47唯一父提交为W47、仅改变Phase47结果目录、manifest完整、五模型均PASS、总请求225且精确匹配225、DeepSeek为TRTLLM MLA/page64、其余四模型为FlashInfer/page1、raw/权重/token均不在Git。全部通过后才能把R47以ff-only方式合入正式分支并push；该新HEAD才可成为Phase48的workflow父结果。
