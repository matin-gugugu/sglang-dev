# Phase 32D：TP绝对上限内的定向gate救援

Phase32C一次性新确认显示TP接近calls有条件线但cost仍失败。本阶段只复用Phase32B的开发OOF residual和三个seed checkpoint，比较global、policy、model、phase、model×policy、policy×phase六种有界gate，使TP累计配置达到绝对上限48。gate、alpha和候选选择没有读取Phase32C或Phase31D target。

开发OOF选择`tp32_rescue_policy_phase_alpha1.0`；其calls/bytes/TV/EMD/cost WAPE分别为8.5180%/2.3917%/0.1616/0.0182/5.0654%。冻结预测SHA为`95f7c5fb505a5cebe72514659e8946ab1b75cd2f5a0635c349b02c30967fba02`。

由于确认target已经在Phase32C开放，下一步在新确认和原固定集上的结果都只能称为重复工程证据；不构成新的盲测。
