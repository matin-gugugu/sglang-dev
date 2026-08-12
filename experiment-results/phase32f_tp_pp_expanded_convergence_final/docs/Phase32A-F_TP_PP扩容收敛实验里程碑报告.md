# Phase 32A-F里程碑归档

1. Phase32A冻结扩容合同与9个新请求级互斥BurstGPT窗口：PASS，target未生成。
2. Phase32B完成TP 42组、PP 30组常规上限搜索并先冻结预测：PASS。
3. Phase32C在冻结SHA后一次性生成新Hfull teacher并评测：TP fail，PP conditional_pass。
4. Phase32D用开发OOF完成TP最后6组gate救援：PASS，累计48组绝对上限。
5. Phase32E只评估已冻结救援预测：TP仍fail；明确标为重复工程证据。
6. Phase32F停止训练并完成总归档：TP未收口，PP有条件收口。

数据量：新确认9个画像、2,976个完整teacher请求、972条target phase rows；Phase32B冻结4,104条预测phase rows，Phase32D冻结2,052条TP预测phase rows。目录体积以最终manifest和`du`复核为准。

提交链：Phase32A=`b476e093`，Phase32B=`89b91d9b`，Phase32C=`0a8b5d74`，Phase32D脚本/结果=`067e890c`/`842479a7`，Phase32E结果=`c1f73c6f`；Phase32F提交以最终Git HEAD为准。
