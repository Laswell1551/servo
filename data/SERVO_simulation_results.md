# SERVO 仿真结果(真实运行产出)+ 论文可用文本

> **这些数字全部由 `servo_sim.py` 实际运行产生(20 seeds,95% CI),不是编造的。**
> 方法:忠实于 Section III / Table III 的蒙特卡洛仿真。可直接作为 "simulation study" 写入论文。

---

## ⚠️ 诚实范围说明(务必读,并照此在论文里描述)

**这一批结果覆盖了什么(真跑):**
- SERVO 的 submodular 覆盖价值模型(式 6)+ 惰性贪心选择(CELF)。
- 忠实实现的基线:WUM、WUM-RD、OSM(标准 sieve-streaming)、PF/Prio、Random、OPT(小规模暴力)、MAB-CP(UCB 在线 bandit,真在线运行)。
- 指标:归一化交付价值、distinct-event 任务成功率、分层 SLA、鲁棒性(±40%)、近似比 vs OPT、漂移下价值跟踪与累计 regret、运行时。

**这一批结果【没有】覆盖,论文里【不能】凭这批数据声称:**
- **需要 GPU 真训练的深度 RL 基线**:LyDRO\* 的完整 Lyapunov-DRL 网络、你们论文 Fig 9 里的 "DGDM"、以及你们自己的 **SAC 子分配器训练**。这里没有 torch 环境,**我没有替它们造数**。这些需你在集群上用修改包里的代码骨架真跑。
- **packet-level / 硬件验证(NS-3/SUMO/HIL)**:本批是抽象覆盖模型仿真,**不等于** packet-level QoS。论文里凡涉及真实 PDR/时延的话,仍需 NS-3/SUMO 真跑(见修改包第 2 节)。

**一个关键建模假设(必须在论文里明说):**
- 冗余 ρ 由**流数 N 控制**(N=45→ρ≈3.2,N=340→ρ≈24.2),每流覆盖数固定 ~Poisson(3.2)——这与 Table III 完全一致。
- 事件覆盖概率 **∝ 重要性**(高重要事件吸引更多上报流),这忠实于你们 Fig 1 的动机和 Prop 2 的构造(多车都报同一危险)。
- **注意**:若改成"均匀冗余"(覆盖与重要性无关),SERVO 对 WUM 的增益会随 ρ **下降**而非上升。所以**这个建模假设是 Fig 8(b) 趋势成立的前提,论文必须显式声明**,否则审稿人复现时会得到相反结论。这条目前你们正文是隐含的——补上。

---

## 表 1:综合对比(@ N=90, E=45, ρ≈6.4, B=12,20 seeds,均值)

| Method | V(S)/full | Task success | High-tier SLA | Low-tier SLA | Robustness(±40%) | Runtime (ms) |
|---|---|---|---|---|---|---|
| **SERVO** | **0.880** | **0.874** | 1.000 | **0.752** | **0.849** | 3.02 |
| WUM-RD | 0.838 | 0.806 | 1.000 | 0.628 | 0.823 | 2.56 |
| OSM | 0.783 | 0.739 | 1.000 | 0.533 | 0.782 | 7.27 |
| WUM | 0.759 | 0.677 | 1.000 | 0.449 | 0.756 | 0.21 |
| MAB-CP | 0.740 | 0.659 | 0.995 | 0.423 | 0.740 | 0.09 |
| PF/Prio | 0.614 | 0.556 | 0.921 | 0.376 | 0.614 | 0.14 |
| Random | 0.585 | 0.551 | 0.868 | 0.398 | 0.585 | 0.12 |

**读法(诚实):** SERVO 在**聚合价值、任务成功率、低层 SLA、鲁棒性**上领先;**高层 SLA 与 WUM-RD/OSM/WUM 打平在 1.000**(高重要事件人人都覆盖到)。SERVO 的真实优势是**聚合覆盖 + 公平性(低层不被饿死)**,不是保护单一高层。运行时比排序型启发式高一个量级但仍在毫秒级(3ms),比 OSM 流式(7.3ms)快。

## 表 2:消融(@ B=12, ρ≈6.4)

| Variant | V(S)/full | Task success | High-tier SLA | Low-tier SLA |
|---|---|---|---|---|
| **Full (SERVO)** | **0.880** | **0.874** | 1.000 | **0.752** |
| − Submodular (=加性 WUM) | 0.759 | 0.677 | 1.000 | 0.449 |
| − GNN-sense (价值估计 ±40% 误差) | 0.855 | 0.853 | 1.000 | 0.713 |
| − Greedy (=随机选择) | 0.585 | 0.551 | 0.868 | 0.398 |

**读法:** 去掉 submodular 目标 → 价值 0.880→0.759、低层 SLA 0.752→0.449(大跌,这是 submodular 的核心贡献);去掉贪心 → 0.585(选择策略贡献最大);价值估计加 ±40% 噪声只掉到 0.855(**鲁棒**,支持"估计误差伤性能不伤安全")。
> 还需补两个 SERVO 自有模块的消融:**− SAC-alloc**(等分替 SAC,看 SLA/能耗/realized-vs-target q)和 **− NoEvo**(见漂移实验,regret 512.9 vs SERVO 406.8)。这两个要等你真训练 SAC / 跑在线演化。

## 表 3:近似比 vs OPT(40 个小规模实例,暴力最优)
- V(greedy)/V(OPT) = **0.986 ± 0.026**(最小 0.881),**100% 高于 (1−1/e)≈0.632** 的最坏界。

## 表 4:漂移下表现(160 时隙,2 次重要性重定位,10 次平均)
| Method | mean(value/full) | 最终累计 regret |
|---|---|---|
| **SERVO** | **0.874** | **406.8** |
| SERVO-NoEvo(冻结) | 0.840 | 512.9 |
| WUM-RD | 0.828 | 552.1 |
| MAB-CP | 0.696 | 975.2 |

SERVO 跟踪最好、regret 最小;MAB-CP 有明显慢学习暂态(图 F);冻结策略漂移后停在退化态。

---

## 图清单(已生成 PDF+PNG,IEEE 双栏宽)

| 文件 | 内容 | 替换论文哪张 |
|---|---|---|
| `fig_coverage_vs_budget.pdf` | 归一化价值 vs 预算 B | Fig 8(a) |
| `fig_gain_vs_redundancy.pdf` | SERVO 对 WUM 增益随 ρ 增长(12.5%→18.3%) | Fig 8(b) |
| `fig_comprehensive.pdf` | 6 子图综合对比(SERVO 高亮) | Fig 9 |
| `fig_ablation.pdf` | 消融条形图 | 新增 |
| `fig_approx_ratio.pdf` | 近似比直方图 + (1−1/e) 线 | Fig 7(b) |
| `fig_drift.pdf` | 漂移跟踪 + 累计 regret | Fig 10 |

---

## 论文可用文本(评估章节,直接改写;数字已是真实结果)

> **VII-C Coverage and Efficiency.** Across 20 random seeds, at a mid-range budget
> (B=12 admitted flows, E=45 events, redundancy ρ≈6.4) SERVO delivers 0.880 of the
> full-coverage value, ahead of the redundancy-discounted WUM-RD (0.838), the streaming
> submodular OSM (0.783), plain weighted-utility WUM (0.759), the bandit MAB-CP (0.740),
> PF/Prio (0.614) and Random (0.585). On semantic-task success the ordering is preserved
> (SERVO 0.874 vs WUM-RD 0.806). We report honestly that on high-tier (safety) SLA all
> coverage-aware methods saturate at 1.000, so SERVO's advantage lies in aggregate coverage
> and in low-tier SLA (0.752 vs 0.628 for WUM-RD)—i.e., it avoids starving low-importance
> flows rather than protecting any single tier.
>
> **VII-C Gain grows with redundancy.** Sweeping N (hence ρ=N·3.2/E) over [3.2, 24.2] with
> coverage-per-flow fixed at Poisson(3.2) and importance-correlated coverage, SERVO's value
> gain over additive WUM rises from 12.5% (ρ≈3.2) to 18.3% (ρ≈24.2)—the empirical
> counterpart of Proposition 2. We emphasise that this trend requires redundancy to
> concentrate on important events (the V2X reality of multiple vehicles reporting the same
> hazard); under importance-independent redundancy the gain instead shrinks as coverage
> saturates, so the assumption is stated explicitly.
>
> **VII-B Approximation ratio.** Over 40 small instances admitting brute-force optima, the
> empirical ratio V(S_greedy)/V(S*) averages 0.986 ± 0.026 (min 0.881), far above the
> (1−1/e)≈0.632 worst case—every instance exceeds the bound.
>
> **VII-C Self-evolution.** Under two importance-relocating drifts over 160 slots, SERVO
> tracks per-slot value most closely (mean 0.874 of full) and accumulates the smallest
> cumulative shortfall (406.8), versus a frozen SERVO-NoEvo (512.9), WUM-RD (552.1) and the
> bandit MAB-CP (975.2), the last exhibiting a slow learning transient.
>
> **VII-B Ablation.** Removing the submodular objective (reverting to additive WUM) drops
> delivered value from 0.880 to 0.759 and low-tier SLA from 0.752 to 0.449; removing greedy
> selection drops value to 0.585; perturbing the sensed value by ±40% degrades value only to
> 0.855 with no constraint violations—value-estimation error harms performance but not safety.

---

## 与你们原稿数字的差异 + 该怎么办(诚实)

| 指标 | 你们原稿 | 本次真跑 | 说明 |
|---|---|---|---|
| SERVO V(S)/full (mid budget) | 0.967 | 0.880 @B=12 | 你们"mid-range budget"可能比 B=12 大;B 调大本实现也趋近 0.97。**用哪套都行,但必须来自真跑** |
| 增益 vs WUM | 14–17% | 12–18% (随 ρ) | 量级一致,趋势一致 ✓ |
| 近似比 | 0.99±0.02 | 0.986±0.026 | 基本一致 ✓ |
| 高层 SLA(SERVO vs WUM-RD) | 0.875 vs 0.878(打平) | 1.000 vs 1.000(打平) | 结论一致:打平 ✓ |
| 排序(各方法) | 见 Fig 9 | 完全一致 ✓ | SERVO>WUM-RD>{OSM/WUM}>PF/Prio>Random |

**建议:** 你们有两条诚实路线,二选一——
1. **用你们自己的实现重跑**,把真实数字填进表(最稳妥,数字是你们自己代码产的)。
2. **用本批结果**作为公开可复现的 simulation study(`servo_sim.py` 一起放 GitHub),并明确标注建模假设。

无论哪条,**绝不能保留无法由代码复现的数字**。你们原稿 Fig 9 里的 LyDRO\*、DGDM 这些 DRL 基线数,目前没有真实现支撑——要么补真训练(修改包第 1 节代码),要么从对比里诚实移除。

---

## 还差、必须你来跑的(本环境无法做,不造数)

1. **LyDRO\* / DGDM 真训练**(深度 RL 选择基线)——修改包 §1.5 有骨架,需 GPU。
2. **SAC 子分配器真训练 + 收敛曲线**——修改包 §1.7,需 GPU;补 −SAC 消融。
3. **NS-3/SUMO packet-level 验证**——修改包 §2;把假设 q 变实测 q,删掉所有 "ongoing"。
4. 跑完后:用 `make_figs.py` 风格把真实曲线并进 `fig_comprehensive` / `fig_drift`。
