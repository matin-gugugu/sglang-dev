# Phase 30B：TP结构化batch事件监督数据集

状态：**PASS**。本阶段把Phase 29允许复用的42个开发画像与Phase 30A的90个全新画像合并，
形成75 train、27 validation、15 first confirmation和15 second confirmation，共132个独立画像。
训练单位是画像×固定策略，不再按模型、TP size或phase重复计算独立样本。

每个单位保存91列低维画像/策略特征、compact32 H0的62维结构化事件先验。开发集306个单位
含Hfull event target，其中225用于拟合、81用于早停；第一确认45个feature与45个target分文件；
第二确认仅45个无target feature，Hfull event target尚未生成。

62维目标由23个prefill batch-count、23个prefill token-mass和16个decode active-lane step
count组成。结构适配器对Phase 29既有标签和Phase 30新teacher共做6,318次Hfull与7,128次H0
跨模型/TP展开核验，所有12桶calls与logical-bytes误差均在浮点容差内。因此DNN只需学习
scheduler事件，模型collective数与bytes/token由确定性适配器恢复。

90个新窗口共71,967个完整请求。请求数组只在内存中用于低维聚合、H0和离线
teacher，没有保存到profiles、labels、dataset或Git。Phase 29两批确认画像未进入本数据集；
Phase 30两批确认预测必须在读取第一确认target前同时冻结。
