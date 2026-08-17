# Phase44：扩展PD开发集与H0保护残差训练

状态：`PASS`。从1200个互不重叠且避开历次窗口的BurstGPT画像生成标签，共486242个完整请求；完整请求未进入Git。

选中`pd44_causal_w32_d1`、alpha=`0.5`。OOF gate=`True`，240画像validation overall gate=`True`，三segment gate=`True`，最终model_accepted=`True`。只有最终为true才允许新blind。
