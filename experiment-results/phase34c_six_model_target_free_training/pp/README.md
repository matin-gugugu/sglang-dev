# Phase 34C-PP：六模型H0+DNN residual训练与预测冻结

本方向使用固定94个开发画像、35,524个唯一完整teacher请求和六个模型，共10,152条phase样本。全部候选都做profile-grouped五折：同一画像派生的六模型、并行配置、policy和phase始终在同一折。Phase34新确认target与Phase33重复集target均未读取。

常规有限搜索18组，每组1个seed初筛，前三名做3-seed × 5-fold确认。选中`pp34_c04_pp_split_retrain_policy_lr0.003_w64_5fold_3seed_alpha1.0`，保留非零DNN residual；开发侧calls/bytes/TV/EMD/cost WAPE为`4.90%`、`0.00%`、`0.1432`、`0.0190`、`3.46%`。六模型bytes结构锚点最大相对误差为`1.130e-15`。

已对Phase34的12个全新确认画像和Phase33的9个已打开重复画像冻结H0及H0+DNN预测。冻结文件SHA-256为`4fc00d2f2378c065f29db74eb5c138455b99f4727802888a1eb4b71ab6e320bb`；只有归档本结果后才能生成Phase34新确认Hfull target。
