# Phase63资源分配澄清：峰值只需要2台node

## 一句话结论

Phase63任意时刻只运行一个measurement shard。L1使用1台node，L2/L3使用2台node，因此全局峰值是2台node、3个GPU进程。禁止申请4台node并发运行，也禁止并行两个measurement shard。

## 四个slot为什么不是四台node

一套placement预先登记`A0/A1/B0/B1`四个GPU slot，是为了让P1D2和P2D1先后复用同一拓扑：

~~~text
L2/L3的两台node：

node A：A0、A1
node B：B0、B1

先运行P1D2：A0 + B0 + B1，共3个GPU进程
结束并释放进程
再运行P2D1：A0 + A1 + B0，共3个GPU进程
~~~

第四个slot在任一具体shard中都不启动。四个slot表示两台node上的四个GPU位置，不表示四台node。

## 两个replica为什么也不增加峰值node数

replica0和replica1是两次独立测量，不是两套同时运行的服务。它们必须顺序执行，可以在相同node/node pair上使用不同GPU tuple：

~~~text
allocation 1：运行replica0的所有需要项 → 完成
allocation 2：运行replica1的所有需要项 → 完成
~~~

如果同一对8-GPU node能够提供两套不同GPU tuple，可以一直复用这两台node；不需要为replica再增加两台node。

## 为什么完整inventory里可能出现超过2个host

L2要求两台node位于同一rack，L3要求两台node位于不同rack。同一node pair不能同时既是同rack又是跨rack，因此完整实验期间可能先后使用不同的node pair。例如：

~~~text
第一批allocation：node A + node B（同rack），完成全部L2 shard后释放
第二批allocation：node A + node C（跨rack），完成全部L3 shard后释放
~~~

这个例子在inventory里出现A/B/C三个host，但峰值仍只有2台。若资源环境更方便，也可以L2使用A/B、L3使用C/D；四个host仍然分成两个顺序allocation，而不是一次申请四台。

## 合规调度顺序

下面任一种顺序都可以，只要一次只有一个shard：

1. 按拓扑：先全部L1，再全部L2，最后全部L3；不同拓扑使用独立allocation。
2. 按资源窗口：拿到一台node时完成L1；拿到同rack node pair时完成L2；拿到跨rack node pair时完成L3。
3. replica0和replica1在同一node pair上换GPU tuple并顺序运行。

模型、P1D2/P2D1、replica、repeat和拓扑之间都不允许并行。唯一允许的并发是一个shard内部的两条PD传输，以及对应的3个rank进程。

## 必须停止并纠正的错误规划

- 因为看到`A0/A1/B0/B1`而申请4台node；
- 因为看到两个replica而同时申请两对node；
- 为了加速而并行运行两个measurement shard；
- 要求一个scheduler allocation同时包含L1/L2/L3的全部host；
- 将inventory累计host数量报告成峰值同时node数量。

这些都属于资源规划误读，不是实验BLOCKED。正确做法是拆成最多2台node的顺序allocation。
