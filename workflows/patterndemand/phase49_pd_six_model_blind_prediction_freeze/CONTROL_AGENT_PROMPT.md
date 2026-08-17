# Phase49 控制端合同

只在 exact W49、除 `data/` 外干净的正式分支执行。先验证 R48 result commit 与全部 pin、selection 可重现、历史 embargo、六份 raw 哈希和 R48 checkpoint 身份。不得训练、调参、计算评价指标或生成 Hfull。

结果只选择性添加 Phase49 allowlist；被 `.gitignore` 忽略的合同 CSV/log 必须逐文件 `git add -f`，禁止 `git add .`。R49 必须唯一父提交 W49，路径和 commit 内 manifest 通过 `verify_result_commit.py` 后才能 ff-only 合入。合入后才允许制作 W50。
