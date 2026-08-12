# Phase 28A：第二批PP独立确认合同

本阶段在生成任何Phase 28预测和Hfull标签前，冻结Phase 27D确认后形成的方法映射：
`MB1=H0、MB4/MB16=增强bounded residual`。从Phase 15窗口中排除Phase 16的24个窗口和
Phase 27的60个窗口，再用相同49个历史侧特征按3/3/3/4/4/1配额
选择medoid，共18个第二独立确认画像。Mooncake synthetic总共只有12个候选，排除前两轮后
仅剩1个，因此将多出的2个配额分给conversation和toolagent；该调整发生在任何选择清单、
预测或Hfull标签生成之前。

`selection/selected_windows.csv`是不可事后更改的窗口清单；
`frozen_method_mapping.json`是不可事后更改的方法映射。当前没有Phase 28预测或Hfull标签，
因此后续可以为这份混合映射提供真正独立的成绩。
