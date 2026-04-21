# 语音版 DiffAttack 双目标实验设计与展示建议

## 1. 这份文档要解决什么

你现在想做的不是一个普通的 speaker attack，而是一个 **purification-aware 的语音保护扰动**：

- 在原始音频上加入微小扰动后，先让说话人特征失效
- 即使攻击者先做一次语音净化，扰动效果也尽量不要被洗掉
- 这个“说话人失效”既可以做成 `untargeted`，也可以做成 `targeted`
- 你手上已经有两条净化路线：
  - 频谱图上做净化
  - 波形上做净化

因此，最自然的论文主线不是“又做了一个 speaker PGD”，而是：

> 参考 DiffAttack 的中间轨迹偏移思路，在语音场景中设计一个同时破坏 speaker identity 和 purification recovery 的双目标攻击框架。

我建议把这条线内部命名为：

- `Audio-DiffAttack`
- 或者如果你想跟论文草稿统一，也可以继续叫 `AntiPure`

## 2. 最值得回答的 4 个研究问题

### RQ1. 只做说话人失效，为什么不够

需要证明：

- 普通 speaker-only 攻击在净化前有效
- 但经过 waveform purifier 或 spectrogram purifier 后，效果明显回退
- 你的方法在净化后仍然保持较高攻击成功率

这是整篇工作的必要前提。

### RQ2. 语音版 DiffAttack 的关键收益是什么

需要证明：

- 只看最终净化输出的 loss 不够
- 在中间 diffusion / denoising 轨迹上施加偏移目标，更容易让净化路径失稳
- 这种“轨迹级 supervision”比 speaker-only 和 two-stage baseline 更强

### RQ3. 波形域和频谱域到底谁更有效

需要回答：

- 在 waveform purifier 下，wave attack 是否更强
- 在 spectrogram purifier 下，spec attack 是否更强
- 两者是否能跨域迁移
- 混合攻击是否优于单域攻击

### RQ4. targeted 和 untargeted 的表现差别是什么

需要区分：

- `untargeted` 更容易优化，通常更稳定
- `targeted` 更难，但更强，更适合体现“控制能力”
- 净化之后 targeted 成功率是否比 untargeted 掉得更快

如果这 4 个问题都答清楚，整篇实验就会非常完整。

## 3. 统一问题定义

设：

- `x` 为干净语音
- `x_adv = clip(x + delta)` 为加扰后的保护语音
- `f(.)` 为说话人编码器
- `P_wav(.)` 为波形域净化器
- `P_spec(.)` 为频谱域净化器
- `e_src = f(x)` 为源说话人 embedding
- `e_tgt` 为目标说话人 embedding

### 3.1 Untargeted 说话人失效

目标是让保护后语音远离源说话人：

`L_spk^untgt = -cos(f(x_adv), e_src)`

如果净化后也想保持失效，可以再加：

`L_post^untgt = -cos(f(P(x_adv)), e_src)`

这里的 `P(.)` 可以是 `P_wav(.)`、`P_spec(.)`，或者两者一起。

### 3.2 Targeted 说话人失效

目标是让保护后语音靠近目标说话人，同时远离源说话人：

`L_spk^tgt = cos(f(x_adv), e_tgt) - lambda_src * cos(f(x_adv), e_src)`

净化后版本：

`L_post^tgt = cos(f(P(x_adv)), e_tgt) - lambda_src * cos(f(P(x_adv)), e_src)`

实践上建议先从下面这个更稳的版本开始：

- 第一阶段先做 untargeted
- 第二阶段再加 targeted attraction

因为直接做 full targeted 往往会更不稳定。

## 4. 参考 DiffAttack 后，语音版最该保留的东西

DiffAttack 真正有价值的不是“PGD”本身，而是下面三点。

### 4.1 中间时间步 supervision

不要只盯着最终 purified waveform，而要在 diffusion 的多个中间时间步上施加偏移目标。

图像版 DiffAttack 结论里很关键的一点是：

- 只在初期时间步加 loss 不够
- 只在后期时间步加 loss 也不够
- **均匀采样多个时间步** 通常最好

语音版建议直接照搬这条经验。

### 4.2 处理随机性的 EOT

净化过程带随机噪声，所以单次 purification 的梯度不稳定。

建议：

- debug 阶段：`EOT = 1`
- 正式实验：`EOT = 3 / 5 / 10`
- 结果至少报告：
  - mean success
  - worst-case success
  - std

否则很容易被 reviewer 质疑“是不是只对某个随机种子有效”。

### 4.3 分段反传 / 内存友好反传

如果你要走 waveform diffusion 路线，完整反传会很吃显存。

你现在仓库里已经有两个很好的切入点：

- `experiments/dual_target_attack/attacks/dual_pgd.py`
  - 适合先验证想法是否有效
- `experiments/dual_target_attack/attacks/diffattack_pgd.py`
  - 更接近 DiffAttack 的 memory-efficient 版本

建议论文里把这两者分工写清楚：

- `DualTargetPGD`：概念验证
- `DiffAttackPGD`：可扩展实现

## 5. 你这篇工作最推荐的 5 个方法变体

为了让实验有层次，建议至少保留下面这些方法。

### M0. Clean

- 不加扰动
- 只作为 speaker similarity 和净化保真度参考

### M1. Speaker-only

- 只优化 `L_spk`
- 不考虑净化
- 这是必须要有的最重要 baseline

### M2. Post-purification only

- 只优化最终净化后输出的 speaker loss
- 不看中间轨迹

这个 baseline 很关键，因为它能证明：

- “只是把净化器串进 loss” 不够
- 真正有效的是轨迹级偏移

### M3. Wave-DiffAttack

- 在 waveform purifier 的中间轨迹上施加 loss
- 这是当前最贴近你现有 `diffattack_pgd.py` 的版本

### M4. Spec-DiffAttack

- 在 spectrogram purifier 的中间轨迹上施加 loss
- 用来证明方法不依赖单一表示空间

### M5. Hybrid-DiffAttack

- 同时利用 waveform loss 和 spectrogram loss
- 可以作为你的 strongest model

如果实验预算有限，最小可发版本可以只保留：

- `M1 Speaker-only`
- `M2 Post-purification only`
- `M3 Wave-DiffAttack`
- `M4 Spec-DiffAttack`

Hybrid 可以作为加分项。

## 6. 波形域和频谱域分别怎么定义 loss

## 6.1 波形域版本

如果净化器本身就在 waveform 上扩散和反扩散，建议主损失写成：

`L_total = alpha * L_spk + beta * L_traj_wav + gamma * L_post`

其中：

- `L_spk`：净化前 speaker 失效
- `L_traj_wav`：中间时间步的 forward / reverse 偏移
- `L_post`：净化后的 speaker 失效

最简单可用的 `L_traj_wav` 形式有两种：

1. 直接做 waveform/state 的 MSE 偏离
2. 做 speaker embedding divergence

从你现在的目标来看，我更推荐第二种，因为你真正关心的是身份信息，而不是纯粹的样本重建误差。

可以写成：

`L_traj_wav = Avg_t [1 - cos(f(x_t^fwd), f(x_t^rev))]`

如果数值不稳定，再退回：

`L_traj_wav = Avg_t ||x_t^fwd - x_t^rev||_2`

## 6.2 频谱域版本

如果你在 Mel / STFT / LogFBank 上做净化，建议别只做 feature L2，要尽量和 speaker objective 接起来。

推荐三档方案：

### 方案 A. 最容易先跑通

- 中间时间步用 `L1/L2` 偏离频谱
- 最终输出恢复到 waveform 后，再算 speaker loss

即：

- `L_traj_spec = Avg_t ||S_t^fwd - S_t^rev||_1`
- `L_post` 在重建回 waveform 后计算

### 方案 B. 更合理

- 频谱净化器输出的每一步都尽量映射回 speaker-related space
- 可以借助 vocoder 或 ISTFT 重建，再算 embedding 偏移

即：

`L_traj_spec = Avg_t [1 - cos(f(vocoder(S_t^rev)), f(vocoder(S_t^fwd)))]`

### 方案 C. 最强但最复杂

- feature deviation + speaker deviation 一起做

即：

`L_traj_spec = mu * feature_gap + nu * speaker_gap`

如果你想尽快出结果，建议先跑 A，再补 B。

## 7. 我最建议的实验矩阵

不要一上来把所有组合全铺满，否则很容易跑崩。建议分三层。

## 7.1 第一层：最小可验证实验

只回答“联合优化是不是比 speaker-only 更抗净化”。

固定：

- 数据集：先小规模 debug 集，再上 VoxCeleb1 test subset
- speaker encoder：先用当前 ECAPA-TDNN
- purifier：先用现成 waveform purifier
- 任务：先做 `untargeted`

比较：

- Clean
- Speaker-only
- Post-purification only
- Wave-DiffAttack

指标：

- 净化前 untargeted success
- 净化后 untargeted success
- source similarity before / after purification
- SNR / PESQ / STOI

只要这层结果成立，你的主线就已经站住了。

## 7.2 第二层：方法完整性实验

把另外两条主轴补齐：

- `untargeted` vs `targeted`
- `waveform purifier` vs `spectrogram purifier`

建议的二维矩阵：

| 攻击方法 | 说话人目标 | 测试净化器 |
| --- | --- | --- |
| Speaker-only | untargeted | wave |
| Speaker-only | untargeted | spec |
| Wave-DiffAttack | untargeted | wave |
| Wave-DiffAttack | untargeted | spec |
| Spec-DiffAttack | untargeted | wave |
| Spec-DiffAttack | untargeted | spec |
| Wave-DiffAttack | targeted | wave |
| Spec-DiffAttack | targeted | spec |

如果资源足够，再加：

- Hybrid-DiffAttack

## 7.3 第三层：论文加分实验

这部分最能体现你“真的参考了 DiffAttack”，而不是只借了名字。

建议做：

1. 时间步采样策略消融
   - 只在 early steps 加 loss
   - 只在 late steps 加 loss
   - uniform sampled steps

2. diffusion length 消融
   - 小步数
   - 中等步数
   - 大步数

3. EOT 消融
   - 1
   - 3
   - 5
   - 10

4. loss 权重消融
   - `alpha / beta / gamma`

5. capture stride 消融
   - 稀疏时间步
   - 稠密时间步

## 8. 指标怎么定义才不会乱

语音领域里 `ASR` 很容易被误解成 automatic speech recognition，所以我建议你不要直接把攻击成功率写成 `ASR`。

推荐命名如下。

### 8.1 Untargeted 成功率

- `USR` = Untargeted Speaker-failure Rate
- 条件：`sim(adv, src) < tau_src`

净化后：

- `P-USR` = Post-purification Untargeted Speaker-failure Rate

### 8.2 Targeted 成功率

- `TSR` = Targeted Speaker-failure Rate
- 条件建议不要只看 `sim(adv, tgt)`，而要看：
  - `sim(adv, tgt) > tau_tgt`
  - 且 `sim(adv, tgt) - sim(adv, src) > m`

净化后：

- `P-TSR`

### 8.3 保真和可懂度

至少报告：

- `SNR`
- `PESQ`
- `STOI`

如果你在频谱图上做文章，建议再补：

- `LSD` 或 spectral convergence

### 8.4 净化器副作用

这个指标非常重要，但很多人会漏掉。

你要报告：

- `clean -> purified(clean)` 的 speaker similarity

因为如果净化器本身就严重破坏 speaker identity，那么你的“净化后仍失效”就不够有说服力。

### 8.5 更强的验证指标

如果你后面要往 speaker verification 论文风格靠，建议加：

- `EER`
- `minDCF`
- `TAR@FAR`

并分别报告：

- clean trials
- adv trials
- purified adv trials

这样更像标准 ASV 实验。

## 9. 最推荐的图表组织方式

如果你问“怎么展现效果最好”，我最推荐下面这个顺序。

## 9.1 主表：最重要的一张表

一张大表就回答核心问题：

- speaker-only 净化前有效，净化后失效
- 你的方法净化前有效，净化后仍有效
- wave/spec/hybrid 谁更强

列建议：

- Method
- Objective (`untargeted` / `targeted`)
- Purifier (`wave` / `spec`)
- `USR` or `TSR`
- `P-USR` or `P-TSR`
- source similarity after purification
- `PESQ`
- `STOI`

## 9.2 图 1：方法框架图

画清楚 4 个模块：

- 输入语音
- 扰动优化器
- 净化器
- speaker encoder

并强调：

- 你不是只对最终输出施加 loss
- 你还对中间 diffusion trajectory 施加 loss

## 9.3 图 2：可视化对比图

每个案例放 4 行最合适：

1. clean waveform / spectrogram
2. protected waveform / spectrogram
3. purified(protected) waveform / spectrogram
4. residual difference

左右两列分别放：

- speaker-only baseline
- 你的方法

这样 reviewer 一眼就能看到：

- baseline 的扰动被净化抹平了
- 你的扰动在净化后仍然留下 identity-shifting 结构

## 9.4 图 3：时间步轨迹图

这是最像 DiffAttack 的一张图，建议一定做。

横轴：

- diffusion step

纵轴可以选：

- `cos(f(x_t^fwd), f(x_t^rev))`
- 或 `||x_t^fwd - x_t^rev||`

比较：

- clean
- speaker-only
- 你的方法

如果你的方法曲线在多个时间步都更偏离，就能证明“轨迹偏移”确实发生了。

## 9.5 图 4：trade-off 图

散点图最直观。

横轴：

- `P-USR` 或 `P-TSR`

纵轴：

- `PESQ` 或 `STOI`

不同点：

- 不同方法
- 或不同 `alpha / beta / gamma`

这张图适合说明：

- 你的方法不是靠把音频破坏得特别烂才成功

## 9.6 图 5：跨域迁移热力图

如果你做 wave/spec 两条线，这张图会很加分。

行：

- optimize on wave
- optimize on spec
- optimize on hybrid

列：

- test on wave purifier
- test on spec purifier

格子里放：

- `P-USR` 或 `P-TSR`

这样能很清楚说明：

- 单域攻击是不是过拟合
- 混合攻击是不是更泛化

## 10. 音频 demo 怎么选最有说服力

如果你要做补充材料或答辩展示，我建议每个案例放 4 段音频：

1. clean
2. speaker-only protected
3. speaker-only purified
4. ours purified

每段音频旁边同时放 3 个数字：

- `sim_to_src`
- `sim_to_tgt`
- `purified success`

如果是 untargeted，就重点放：

- `sim_to_src`

如果是 targeted，就重点放：

- `sim_to_tgt - sim_to_src`

最好的 demo 不是“听起来最奇怪”的，而是：

- clean 和 protected 在听感上接近
- 但 embedding 指标和净化后指标明显变化

## 11. 一个很实用的论文叙事结构

如果后面你要把这块写进论文，实验部分可以按这个顺序展开：

1. `Speaker-only` 在净化面前不够鲁棒
2. 引入 post-purification supervision 后有提升，但还不够
3. 引入 trajectory deviation 后，净化后成功率显著上升
4. waveform 和 spectrogram 都能做，但各有偏置
5. hybrid 最稳，或者 cross-domain 泛化最好
6. uniform time-step sampling 和 moderate diffusion length 最合适

这个结构和 DiffAttack 的叙事是对齐的，但又明显是音频版本，不会像简单复刻。

## 12. 最小可落地执行顺序

如果你现在就想往下推进，最推荐按这个顺序做。

### 第一步：先把 untargeted waveform 跑稳

只做：

- Speaker-only
- Post-purification only
- Wave-DiffAttack

先验证：

- 净化前成功
- 净化后仍有差距

### 第二步：补 targeted

先不要一开始就 full targeted。

建议顺序：

1. untargeted 稳定
2. targeted 只加最终 loss
3. targeted 再加 trajectory loss

### 第三步：把 spectrogram 支路补成平行实验

重点看两件事：

- spec attack 在 spec purifier 下是不是更强
- spec attack 能不能迁移到 wave purifier

### 第四步：再做 hybrid 和消融

等前面主结果稳定后，再补：

- hybrid
- EOT
- timestep strategy
- diffusion length

## 13. 我对你当前项目的直接建议

结合你仓库当前状态，我建议你下一步优先做下面几件事。

### 建议 A. 先把“命名和指标”统一

当前代码里已经有：

- `dual_pgd.py`
- `diffattack_pgd.py`
- `metrics.py`

但文档和论文最好统一成：

- `speaker-only`
- `post-purification`
- `wave-diffattack`
- `spec-diffattack`
- `hybrid-diffattack`

同时把攻击成功率命名成：

- `USR / P-USR`
- `TSR / P-TSR`

不要直接写 `ASR`。

### 建议 B. 当前 artifact 先以 ECAPA 为主

你现在代码里主 speaker encoder 是 `ECAPA-TDNN`，所以实验文档、图表和论文先围绕它写最稳。

后面如果要做迁移性，再补：

- x-vector
- WeSpeaker

### 建议 C. 把 waveform branch 作为第一主线

因为你当前仓库里：

- `diffattack_pgd.py`
- `De-AntiFake` purifier

已经比较接近 waveform DiffAttack 了，所以最省力的路线是：

- 先把 waveform 版本做成完整主结果
- spectrogram 版本做成平行验证和扩展

### 建议 D. 图里一定要加 trajectory figure

如果没有这张图，别人很容易觉得你只是“在最终输出上多加了一个 loss”。

有了 trajectory figure，方法创新点会更容易站住。

## 14. 一句话版结论

如果要把这项工作压缩成一句最清楚的话，我建议写成：

> 我们不是只让语音在净化前失去说话人信息，而是参考 DiffAttack，把攻击目标推进到净化轨迹内部，使保护扰动在 waveform 和 spectrogram 两类净化器后仍然保持 speaker disruption，并同时支持 untargeted 和 targeted 两种设置。

## 15. 你现在最值得优先补的结果

如果只能先补三组结果，我建议是：

1. `Speaker-only` vs `Wave-DiffAttack`
   - 看净化前后差距

2. `Wave-DiffAttack` vs `Spec-DiffAttack`
   - 看不同 purifier 下的表现

3. `untargeted` vs `targeted`
   - 看哪一种更稳，哪一种更强

只要这三组结果清楚，这篇工作的实验骨架就已经很扎实了。
