# PatternDemand 图集

本目录由 `generate_patterndemand_figures.py` 从冻结实验产物生成。数据对应分支
`experiment/pattern-demand-v0.5.15-clean`、提交 `ffb413ffec69fd2f87bc958ed73f618696457baa`。

## 图表

1. `01_blind_accuracy_overview`：TP/PP/PD fresh-blind 四个对齐指标的 DNN/H0 误差比例。
2. `02_six_model_robustness_heatmap`：六模型 composite ratio；TP/PP 为 Phase34D 派生口径，PD 为 Phase50 官方口径。
3. `03_error_distribution_ecdf`：单画像 calls histogram TV 的 ECDF。
4. `04`–`09`：TP/PP/PD 各两组 12-bin 样例；每组包含三个画像，并同时展示 calls 与 logical bytes。
5. `10_tp_pp_physical_curves`：Phase39 TP2/TP4/TP8/PP 的 L1/L2/L3 曲线。
6. `11_pd_physical_curves`：Phase51 六模型纯 PD 的 L1/L2/L3 曲线。

每张图同时提供 PNG 和 SVG。`contact_sheet.png` 用于快速浏览。

## 样例选择规则

- 固定 Qwen3-8B；TP 固定 TP4/balanced，PP 固定 PP4/mb4，PD 固定 P1→D1。
- `消息尺度`组：先用固定的 Hfull calls 归一化熵下限排除单桶尖峰（TP/PD 0.25，PP 0.16），再按 12-bin 加权中心取最小/中位/最大画像。
- `分布宽度`组：按 Hfull calls 的加权 bin 标准差，取 90%/50%/10% 分位附近的唯一画像。
- 选择过程只读取 Hfull，不读取 H0 或 DNN 的预测误差。完整选择记录见 `audit/sample_selection.csv`。
- 横轴仅保留该组三个 Hfull 画像在 calls 或 logical bytes 中实际出现过的连续桶范围。

## 口径说明

- 图 01 使用可对齐的 total calls WAPE、total bytes WAPE、calls histogram TV 和 normalized log-payload EMD。
- 图 02 的 composite 仅表示“相对各自 H0 的改善比例”。TP/PP 与 PD 的官方评估阶段对 histogram WAPE 的可用字段不同，因此不应把不同列当成绝对误差横向比较；具体定义保存在 `audit/model_composite_metrics.csv`。
- 物理曲线不做平滑和单调化。实线为 official knot，阴影为 replica min–max；三角形表示跨 replica spread >25%，空心圆表示 repeat median CV >15%。
- Phase54–57 在该提交中只有 workflow，没有完成的 `experiment-results`，本图集不将它们当作已完成精度结果。

## 复现

```bash
python generate_patterndemand_figures.py --data-dir /path/to/copied/frozen/artifacts
```

需要 Python 3.11+、NumPy、pandas、Matplotlib 和 Pillow。脚本不会训练模型或修改原始数据。

## 验证

生成时执行了 schema、行数、profile/method/phase 覆盖、12-bin 长度、曲线唯一性及输出完整性检查。结果见 `audit/validation_report.md`。
