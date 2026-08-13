# Phase34：六模型扩展执行参考

> 本文是 Phase34 的阶段执行约束，不替代《截至目前实验结构总导引》；基础任务、Hfull teacher、fixed-draining 语义、指标定义和 Phase33 已冻结结果均保持不变。

## 1. 阶段目标

在 Phase33 已冻结的三个模型基础上，把模型集合扩展为六个，并使用同一批 94 个开发画像、同一 Hfull teacher 协议，分别重新训练六模型版本的 TP 与 PP `H0 + DNN residual`。Phase33 的 checkpoint、预测、报告和 manifest 只读保存，不修改、不覆盖。

六模型都同时进入训练和验证；不做“留出整个模型”的严格验证。五折划分只按 `profile_id` 进行，同一画像派生的六模型、全部 TP/PP size、policy 和 phase 必须落在同一折，不能跨折泄漏。

## 2. 六个模型及新增选择

保留：

- `deepseek-v2-lite`：较小 hidden size 的 DeepSeek MoE，27 层、hidden size 2048、64 experts、top-6。
- `qwen3-8b`：中等规模 dense，36 层、hidden size 4096。
- `qwen3-30b-a3b`：深层 Qwen MoE，48 层、hidden size 2048、128 experts、top-8。

新增：

- `llama-3.2-3b-instruct`：小型 dense，28 层、hidden size 3072、24 attention heads、8 KV heads。它补充更小参数规模、较浅层数和 6144 bytes/token payload；当前 SGLang 注册模型测试包含该模型。Meta 官方模型卡确认其为 3.21B、128K、GQA；Hugging Face gated 配置不可匿名读取，因此结构字段同时由当前 SGLang 注册记录和公开兼容配置交叉核验，本阶段保存规范化审计配置，不下载权重。
- `qwen2.5-14b-instruct`：较大 dense，48 层、hidden size 5120、40 attention heads、8 KV heads。它补充六模型中最大的 dense hidden size 和 10240 bytes/token payload；当前 SGLang 注册模型测试包含该模型，官方 Hugging Face `config.json` 可直接读取。
- `mixtral-8x7b-instruct-v0.1`：经典稀疏 MoE，32 层、hidden size 4096、8 experts、top-2。它补充与 DeepSeek/Qwen MoE 不同的少专家、低 top-k 结构；当前 SGLang 有直接模型实现和多处测试，官方 Hugging Face `config.json` 可直接读取。

这三个新增模型共同扩大参数规模、dense/MoE、层数、hidden size、KV head ratio 和通信 payload 的覆盖。只读取并固化 KB 级配置；不下载大模型权重，不新增 GPU profiling。

配置来源：

- Meta Llama 3.2 官方模型卡：`https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md`
- Qwen2.5-14B-Instruct 官方配置：`https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/resolve/main/config.json`
- Mixtral-8x7B-Instruct-v0.1 官方配置：`https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/config.json`

## 3. 数据、teacher 与隔离

- 开发数据固定为 Phase33 的 94 个请求级互斥画像：75 个 `development_train`、19 个 `development_validation`，覆盖 90 个 BurstGPT 和 4 个 Mooncake 窗口，共 35,524 个完整 teacher 请求。
- 对同一 94 个画像扩展到六模型。完整请求列表只在离线标签生成时短暂使用，不写入正式数据集，也不进入预测器特征。
- TP teacher 继续使用 Phase26A GPU 验证过的 fixed-draining 结构公式；PP teacher 继续使用 Phase25B/25C 验证过的 scheduler-faithful 事件模拟器。
- TP 与 PP 的 H0、teacher 和最终预测都按每 1000 请求归一化。
- Phase33 已打开的 9 个确认窗口仅是重复工程证据，不进入训练、loss、gate、alpha、checkpoint 或候选选择。
- 从剩余 BurstGPT 历史流量冻结一批全新的、P95 正常范围内、请求级互斥的确认窗口。先只生成低维特征与 H0；训练完成后先归档 checkpoint、冻结预测及 SHA，再一次性生成 Hfull target。打开后不再调参。
- 若累计 300 秒 embargo 下没有足够的新窗口，则只报告开发侧或重复工程证据，不宣称新盲测。

## 4. 六模型训练路线

### TP

- 保留 Phase33 的“总 calls 与 12-bin 形状分头”设计。
- 使用共享主干，并比较 shared、policy、model、model-policy 小头。
- 保留 residual gate、低维顺序/长度/形状特征和由预测直方图计算的 cost 保护 loss。
- bytes 继续使用低维均值结构锚点并保留 H0 bin 形状，但必须在六模型全部开发样本上与 Hfull teacher 逐条核验；不一致就停止直接沿用并进入受限 bytes residual 候选。

### PP

- 不沿用三模型 incumbent 作为最终六模型模型；使用六模型、94 画像的完整开发集重新训练 calls 与形状 DNN。
- 比较 bytes/cost 保护、MB 独立 gate、model/policy 小头与 MB16 保护权重。
- bytes 同样先审计低维均值结构锚点；允许在开发侧受限校准，但不能读取确认 target。

TP、PP 选中模型都必须具有非零 DNN residual，并在 calls 与 common-reference cost 上相对各自 H0 正向改善。

## 5. 有限搜索与停止条件

- TP：常规 18 组，定向救援绝对上限 24 组。
- PP：常规 18 组，定向救援绝对上限 24 组。
- 每组先 1 个 seed；开发侧前三名再做 3 seeds × 5 folds 确认。
- 候选、alpha、gate、checkpoint 的选择只使用开发训练、验证与 profile-grouped CV。
- 达到正式门槛、达到绝对上限、连续两个候选族无改善，或继续必须改变总导引/重新做大规模 GPU profiling 时，停止相应方向。

正式门槛：

- TP：calls WAPE ≤ 10%、bytes WAPE ≤ 2%、TV ≤ 0.20、EMD ≤ 0.025、cost WAPE ≤ 5%。
- PP：calls WAPE ≤ 15%、bytes WAPE ≤ 3%、TV ≤ 0.22、EMD ≤ 0.04、cost WAPE ≤ 5%，并检查逐模型和 MB16 保护条件。
- 六模型均不得明显退化；TP/PP 的 calls 和 cost 都必须相对 H0 改善。

## 6. 归档与保护

每个里程碑保存中文 README、summary、audit、logs、checkpoint、冻结预测、逐模型/逐 policy 指标、图表、DONE 和 manifest。正式结果只选择性 `git add`，禁止 `git add .`；push 到 `experiment/pattern-demand-v0.5.15-clean` 后以 ff-only 同步本地。

继续保护本地 `data/`，远端 Phase16 GPU 旧目录、Phase19 formal-v1/v2/smoke 与 PID、Phase23 旧目录、raw profiler trace、缓存、大模型权重和所有 PID 文件。
