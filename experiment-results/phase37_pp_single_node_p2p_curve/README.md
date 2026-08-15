# Phase37：PP单机P2P真实连续曲线

最终状态：`PASS_WITH_LIMITED_TOPOLOGY`。本阶段在1类实际单机GPU拓扑上测量了`payload → latency`曲线，每条曲线21个payload点，至少5次独立重复。

正式曲线测量的是SGLang `send_tensor_dict(async_send=True)`中GPU tensor对应的NCCL `isend/irecv`原语。每个GPU对双向分别实测，正式拓扑类别曲线按每次repeat的双向较慢值聚合，不能挑选较快方向。它与消息直方图的sender-only logical message口径一致，不包含CPU metadata、tensor分配、scheduler和通信计算重叠。

逐次raw样本保存在Git仓库外；Git只保存紧凑曲线、repeat方差、环境、拓扑、日志、外置raw SHA清单和manifest。曲线保留真实非单调点，不做平滑。Phase38才会将Phase34冻结直方图确定性代入这些曲线。
