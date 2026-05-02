# ACSAC 语音安全相关论文行文逻辑与写作规范

本文档基于 `.pipeline/docs/papers/ACSAC related/` 下 5 篇论文的 `full.md` 阅读整理，用于后续撰写或修改语音安全、ASR 攻防、音频对抗样本、防御检测类论文时参考。

覆盖论文：

- Liu 等，2024/ACSAC'23，`Protecting Your Voice from Speech Synthesis Attacks`
- Park 等，2021/ACSAC'21，`Detecting Audio Adversarial Examples with Logit Noising`
- Schonherr 等，2020/ACSAC'20，`IMPERio: Robust Over-the-Air Adversarial Examples for Automatic Speech Recognition Systems`
- Wang 等，2019/ACSAC'19，`Defeating Hidden Audio Channel Attacks on Voice Assistants via Audio-Induced Surface Vibrations`
- Zhang 等，2021/ACSAC'21，`CommanderGabble: A Universal Attack Against ASR Systems Leveraging Fast Speech`

## 一句话总结

这批 ACSAC 论文的核心写法是：先把语音系统放进真实生活和安全后果中，再指出现有攻防在“真实可部署/真实可攻击”上的缺口，然后提出一个来自声学、物理传输、模型中间表示或语言学误读的核心洞察，把该洞察工程化成攻击或防御系统，最后用多平台、多场景、多指标实验证明它不是玩具例子。

最值得学习的不是它们用了多少公式，而是它们都在回答同一组审稿问题：

- 这个问题为什么现实存在，而不是实验室假设？
- 现有方法为什么还不够？
- 你抓住的关键差异是什么？
- 这个差异为什么难以被攻击者绕过，或为什么足以让攻击成功？
- 你的系统在真实设备、真实房间、真实 ASR 服务、真实用户感知下还能不能成立？

## 通用叙事骨架

### 1. 从普及性进入，而不是从算法进入

这些论文开头几乎都先讲语音交互已经进入现实生活：智能音箱、手机助手、语音认证、在线转写、社交媒体音频、导航、会议、车载场景。这样做有两个目的：

- 让安全问题变成现实风险，而不是模型漏洞。
- 让后续实验场景自然落地到 Alexa、Google Assistant、Cortana、Azure、DeepSpeech、Kaldi、真实房间、真实手机等平台。

可复用写法：

```text
[应用生态] 已经成为 [用户行为/系统入口] 的常见方式。
然而，[输入通道/模型机制/传感器链路] 暴露出一个被低估的安全面：
[攻击者能力] 可以导致 [具体后果]。
```

不要只写“ASR systems are widely used”。要马上接上“攻击会造成什么”，例如打开门、访问钓鱼网站、拨打紧急电话、绕过语音认证、克隆 CEO 声音诈骗。

### 2. 用“现实缺口”定位贡献

这些论文不会只说“已有方法不行”，而是把缺口压缩成非常具体的限制：

- 只能 over-line，不能 over-the-air。
- 需要白盒模型，不适用于商业 ASR。
- 需要特定硬件或特定房间。
- 可以检测数字样本，但检测不了物理播放后的样本。
- 防御有效但牺牲可用性、耗时高、需要逐样本优化。
- 依赖音频域特征，但音频域特征可能被合成或对抗优化伪造。

可复用句式：

```text
Existing work has shown [能力 A]. However, [现实约束 B] remains unresolved:
(i) [限制 1], (ii) [限制 2], and (iii) [限制 3].
These limitations prevent [existing methods] from being deployed in [realistic scenario].
```

写作时要把缺口和本文贡献一一对齐。缺口不是批评清单，而是后文实验设计的蓝图。

### 3. 核心洞察必须是一句“机制差异”

每篇论文都有一个可以单句复述的机制洞察：

- Liu：语音频谱中存在一些频率，修改它们会显著破坏语音合成模型捕捉说话人特征，但对人类感知影响小。
- Park：对 logits 加噪后，良性语音转写稳定，而对抗样本转写易变，因为二者 logit gap 分布不同。
- IMPERio：over-the-air 传输可建模为原始音频与 RIR 的卷积；对一组 RIR 做期望优化可产生跨房间鲁棒样本。
- Wang：隐藏语音命令可以伪造音频域特征，但难以同时伪造由设备结构和传感器采样共同决定的振动域特征。
- CommanderGabble：快速语音会系统性诱发 ASR 的 phoneme reduction/replacement/coalescence，可被反向利用来构造人听不懂但 ASR 执行的命令。

写作规范：核心洞察最好在引言末尾、方法开头、结论中各出现一次，但措辞逐渐变化：

- 引言中：用直觉解释。
- 方法中：用模型、变量、流程或图示落地。
- 讨论/结论中：上升为安全含义。

### 4. 贡献列表的标准顺序

这批论文的贡献列表通常遵循以下顺序：

1. 发现或形式化一个新安全问题/新机制。
2. 设计一个攻击、防御或检测系统。
3. 实现并在真实平台/真实场景中评估。
4. 讨论部署、鲁棒性、局限性或 countermeasure。

更具体地说，attack paper 强调“first/generic/real-world/over-the-air”，defense paper 强调“deployable/low overhead/no retraining/black-box/usable”，detection paper 强调“high accuracy/low FPR/compatible with existing systems”。

贡献列表不要写成模块说明。应写成“本文推进了什么边界”：

```text
We make the following contributions:
- We identify [new vulnerability / new discriminative signal] and explain why it matters under [threat model].
- We design [system/method], which leverages [core insight] to [attack/defend/detect].
- We implement [prototype] and evaluate it across [platforms/devices/scenarios], showing [key result].
- We analyze [robustness/deployability/limitations/countermeasures], demonstrating [practical implication].
```

## 摘要写法

### 推荐结构

这些论文的摘要通常是 5 步：

1. 背景：ASR/语音助手/语音合成已经广泛使用。
2. 风险：该系统暴露某类攻击或滥用。
3. 缺口：已有攻击/防御在现实条件下有明显限制。
4. 方法：提出核心机制和系统。
5. 证据：平台、场景、指标、最亮结果。

示例模板：

```text
[Voice/ASR systems] are widely adopted in [settings].
Recent studies have shown that they are vulnerable to [attack].
However, existing [attacks/defenses] [depend on white-box access / fail over the air / incur high overhead / rely on fragile assumptions].
In this paper, we propose [method], which leverages [core mechanism].
We evaluate [method] on [platforms/devices/datasets/scenarios], and show that [headline result], while [preserving usability / requiring no retraining / working in real-world settings].
```

### 摘要中的数字要选“审稿人能立即理解”的

常见 headline number：

- Park：over-line 100%，over-air >95% detection accuracy。
- Wang：最高 99.9% hidden command detection accuracy。
- CommanderGabble：OTW 96/100 commands，OTA 18 commands across scenarios。
- Liu：攻击成功率显著降低，同时用户感知仍可接受，生成耗时约 1 秒级。
- IMPERio：跨房间、不同距离、无直视条件仍能 over-the-air 成功。

注意：摘要中的数字不是越多越好。选一个“证明主张成立”的数字即可。

## 引言写法

### 引言的 7 段式结构

1. 语音生态普及：谁在用，哪些场景在用。
2. 安全后果：攻击者能造成什么具体危害。
3. 现有工作：已有攻击/防御推进了什么。
4. 关键缺口：为什么仍不现实、不够稳、不易部署。
5. 本文洞察：我们观察到一个可利用/可防御的机制。
6. 本文方案：把洞察转成系统，说明高层流程。
7. 本文贡献：列 3-4 条，最后 roadmap。

写作时，段落之间要形成因果链：

```text
普及性 -> 暴露风险 -> 现有方法不足 -> 本文洞察 -> 系统设计 -> 证据预告
```

### Attack paper 的引言逻辑

以 IMPERio 和 CommanderGabble 为代表：

- 先说明 ASR 对抗样本已经存在。
- 再指出“之前不够现实”：不能 over-the-air、要特定房间、要白盒、要硬件、人能听出目标词。
- 然后提出一个新攻击面：
  - IMPERio：把房间传输纳入优化。
  - CommanderGabble：利用 fast speech 的系统性误读。
- 最后强调“现实强度”：多房间、多距离、无直视、多商业 ASR、家庭/会议/车载场景。

Attack paper 的关键不是“我能攻击”，而是“我比现有攻击更接近真实威胁模型”。

### Defense/detection paper 的引言逻辑

以 Liu、Park、Wang 为代表：

- 先承认攻击有现实危害。
- 再指出现有防御的问题：
  - 检测发生在后验，可能已经造成损害。
  - audio-only 容易被伪造。
  - retraining 或结构改造成本高。
  - time overhead 或 usability cost 高。
- 然后提出“低侵入”的防御信号：
  - 修改特定频段但保持可用性。
  - 在 logits 端加噪，不改模型结构。
  - 利用已有 motion sensor，不加昂贵硬件。
- 最后用 low overhead、no retraining、low FPR、human perception 等指标回应部署问题。

Defense/detection paper 的关键不是“指标高”，而是“部署代价低，误伤低，用户可接受”。

## 方法章节写法

### 先定义威胁模型/问题设定

这类安全论文非常重视 threat model 或 problem setting。它通常回答：

- 攻击者知道什么？
- 攻击者能控制什么？
- 攻击者不能控制什么？
- 输入输出是什么？
- 攻击或防御成功的判据是什么？
- 现实场景包括 over-line、over-air、internal、external、frontend、backend 中的哪些？

建议固定写法：

```text
We consider [black-box/white-box] [attack/defense] setting.
The adversary can [capabilities], but cannot [limitations].
The goal of the adversary/defender is to [objective].
We evaluate success using [metric], where [definition].
```

### 方法不是代码说明，而是“洞察到系统”的转换

优秀写法会经历三步：

1. 观察：展示一个分布差异、频谱差异、语音误读模式、RIR 物理模型。
2. 形式化：给出变量、目标函数、流程图、距离度量、阈值。
3. 系统化：解释如何处理输入、如何搜索/筛选/分类、如何输出攻击或判定。

例如：

- Park 先展示 logit gap 分布差异，再定义 CER 阈值和多次 logit noising。
- Wang 先证明 motion sensor 能采集语音振动，再说明非线性 aliasing 和 feature selection，最后给 supervised/unsupervised classifier。
- Liu 先定义原始语音质量变化和合成语音质量变化，再提出 frequency block、mask 方法和 speaker-level 策略。
- IMPERio 先把房间建模为 RIR，再把 convolution layer 接入 DNN 反传优化。
- CommanderGabble 先分析 fast speech 造成的 phoneme 误读，再设计 phonetic reconstruction、speech synthesis、winnowing/updating。

### 图 1/图 2 要承担“方法入口”

这批论文都用早期图来快速建立心智模型：

- 攻击流程图：normal audio -> perturbation/manipulation -> ASR wrong output。
- 系统总览图：calibration -> feature derivation -> feature selection -> classifier。
- 物理模型图：audio x convolved with RIR h。
- 防御 workflow：speech synthesis attack and defense scheme。

写作建议：

- 第一个图不要太细，目标是让读者知道系统边界和输入输出。
- 方法图 caption 要能独立说明“谁输入、谁输出、为什么有效”。
- 具体算法细节可以放到后续子节，不要在图 1 堆公式。

## 实验章节写法

### 实验不是罗列结果，而是逐个支撑 claim

这些论文的实验都围绕主张展开：

- 有效性：攻击能否成功，防御能否降低攻击成功率，检测能否识别攻击。
- 现实性：是否 over-the-air，是否真实设备，是否多房间/多平台/多场景。
- 鲁棒性：不同距离、不同房间、不同命令长度、不同语音内容、不同设备、不同参数。
- 可用性：正常语音是否仍可用，用户是否听得懂，FPR 是否低，延迟是否可接受。
- 迁移性：是否跨模型、跨平台、跨数据、跨说话人。
- 效率：生成或检测耗时，是否能即时使用。

建议每个实验小节开头都写一句 claim：

```text
This experiment evaluates whether [method] remains effective under [realistic factor].
```

不要直接从“Experimental Setup”跳到表格。表格前后都要告诉读者它证明了什么。

### 实验 setup 的共同要素

需要交代：

- 数据集或命令集合：LibriSpeech、VCTK、常用 voice commands、wake words。
- 平台：DeepSpeech、Kaldi、Google STT、Amazon Alexa/Transcribe、Azure、IBM、Resemblyzer。
- 场景：over-line、over-air、household、teleconference、in-vehicle、frontend/backend、不同房间。
- 硬件：speaker、microphone、smartphone、accelerometer、Raspberry Pi、GPU。
- 指标：accuracy、FPR、FNR、WER、CER、ASR、ACR、SNRseg、human comprehensibility、runtime。
- Baselines：raw/no defense、Attack-VC、RNP、TD1/TD2、Reverb、normal fast speech。

如果论文声称 real-world / over-the-air，物理环境要写细。至少交代房间类型、speaker-microphone 距离、设备型号、播放音量、采样率、背景噪声，能测则报告混响时间 `T60`。这些细节不是装饰，而是支撑“真实可复现”的证据。

### 安全论文特别重视负面指标

Defense/detection 论文必须报告误伤或可用性：

- FPR：把良性语音误判为攻击会破坏用户体验。
- Acceptance rate / human perception：防御处理后的语音还要能正常使用。
- Runtime：如果用于即时语音消息或在线识别，耗时不能过高。
- Partial playback / no retraining / no structural change：证明部署成本低。

Attack paper 必须报告 stealthiness 或 human incomprehensibility：

- 人是否能听懂 adversarial audio。
- 扰动是否低于 hearing threshold。
- 无目标提示时人是否能识别命令。

### Defense 必须考虑 adaptive attacker

防御论文尤其要避免给人“security by obscurity”的感觉。审稿人会自然追问：如果攻击者知道你的防御机制、损失函数、频段选择、检测阈值或净化流程，他能否重新优化绕过？

建议至少在实验或讨论中加入一个 adaptive attacker analysis：

- 攻击者知道防御原理，但不知道私有参数或用户私有样本。
- 攻击者把防御模块近似进优化目标，测试是否还能保持攻击成功率。
- 如果完整 adaptive attack 成本过高，也要说明攻击者需要同时满足哪些互相冲突的目标，例如保留语义、保留音质、绕过检测、维持 speaker similarity。
- 不要只说“adaptive attack is future work”。更好的写法是：给出一个初步 adaptive setting，再说明剩余更强攻击是 future work。

## Related Work 写法

### 不要按论文流水账写

这些论文的 Related Work 通常按方法类别组织：

- White-box audio attacks
- Black-box audio attacks
- Hardware-assisted / ultrasound attacks
- Misinterpretation-based attacks
- Detection methods
- Speech synthesis defenses
- Voice authentication / VCS security

每类都要回答：

```text
这类方法解决了什么？
它们为什么不能解决本文场景？
本文和它们的关键区别是什么？
```

对比句式：

```text
Unlike [line of work], which requires [assumption], our method [removes/relaxes assumption].
Compared with [prior work], our approach targets [more realistic setting] and evaluates [broader scenarios].
```

### 用对比表把 novelty 钉住

安全论文很适合在 Related Work 或 Introduction 末尾放一张 `Comparison of existing attacks/defenses` 表。表格的作用不是堆引用，而是把本文的威胁模型和现实约束摆出来。

常用列包括：

- Threat model：white-box / black-box / gray-box。
- Channel：over-line / over-the-air / real device。
- Domain：time / frequency / model logits / physical vibration / phonetic structure。
- Assumptions：no retraining、hardware-free、room-independent、purification-aware、speaker-independent。
- Evaluation：commercial ASR、human study、adaptive attacker、runtime、utility。

最后一行放本文方法，但不要机械追求“全是 √”。更可信的写法是让表格显示本文解决了最关键的现实缺口，同时诚实保留边界。

### Related Work 也服务于 novelty

CommanderGabble 的 Related Work 明确把自己定位为 fast-speech misinterpretation attack，而不是普通 audio attack。IMPERio 把自己定位为 room-independent robust over-the-air attack。Wang 把自己定位为 vibration-domain defense，而不是 audio-domain classifier。

写作时要避免“我们也做了一个防御”。应该写：

```text
Prior defenses inspect [domain A]. We instead inspect [domain B], because [domain B] is induced by [physical process] and is harder to forge under [threat model].
```

## Discussion / Limitations 写法

这些论文通常把 discussion 写成“主动回应审稿人会问的问题”：

- 是否能迁移到其他 ASR？
- 是否能被 adaptive attacker 绕过？
- 是否只适用于英文？
- 是否要求额外硬件？
- 是否会造成用户打扰或延迟？
- 是否适用于商业黑盒系统？
- 是否可以和其他防御结合？

好的 limitations 不会削弱论文，而是划清边界：

```text
We currently evaluate [scope]. Extending to [larger scope] is future work.
This limitation does not invalidate the main claim, because [reason tied to experiments].
```

### Ethical Considerations 不能缺席

语音攻击、防御、隐私保护和用户实验都容易触及伦理合规。即使会议模板没有强制单独成节，也建议在实验或讨论附近交代：

- Human subject study 是否经过 IRB/伦理审查，参与者是否知情同意。
- 是否采集真实语音、身份相关信息或可识别音频，数据如何匿名化和保存。
- 攻击论文是否限制公开可滥用细节，是否只发布经过降风险处理的 demo。
- 如果涉及真实厂商或商业服务，是否做 responsible disclosure，或说明实验没有造成未授权访问和实际损害。
- 是否评估对普通用户的干扰、误伤、可访问性和隐私影响。

## 逐篇行文逻辑拆解

### Liu: Protecting Your Voice from Speech Synthesis Attacks

论文类型：预防型防御。

主线：

1. 语音合成有正面应用，但也被用于 CEO 声音诈骗、语音认证绕过等真实攻击。
2. 现有 fake speech detection 多是事后检测，且依赖录音条件；Attack-VC 是预防式，但白盒、逐样本优化、效率和可用性不足。
3. 核心洞察：改动特定频率可以破坏合成模型提取说话人特征，同时对人类听感影响小。
4. 方法：定义防御后原语音质量变化和合成语音质量变化；提出 Zero Mask、AN-Mask、GB-Mask；进一步提出 speaker-level universal defense。
5. 实验：VC/TTS 多模型、VCTK、多 SR 系统、用户研究、transferability、efficiency。
6. 结论：防御显著降低攻击成功率，仍保持正常用途，并能约 1 秒级生成防御样本。

可学习点：

- 防御论文要同时证明 security gain 和 utility preservation。
- 引言中用真实诈骗案例极大增强动机。
- speaker-level defense 是把“有效”推进到“可部署”的关键一笔。
- 实验小节安排非常完整：模型、数据、baseline、参数、IRB、主结果、用户研究、迁移性、效率。

### Park: Detecting Audio Adversarial Examples with Logit Noising

论文类型：检测型防御。

主线：

1. ASR 对抗样本包括 over-line 和 over-air，后者更现实。
2. 现有检测方法要么损伤 accuracy，要么只适合 over-line，要么 time overhead 高。
3. 核心洞察：良性样本和对抗样本的 logit gap 分布不同；加噪后，对抗样本 transcription 更不稳定。
4. 方法：在 ASR decoder 前对 logits 加 Gaussian noise，比较原始转写和 noised 转写的 CER，多次采样增强稳定性。
5. 实验：DeepSpeech、3 个 over-line attack、2 个 over-air attack、与 RNP/TD/Reverb 比较，强调 over-air 上的优势和低 FPR。
6. Discussion：解释 logit gap 机制，讨论 noise distribution、不同输入类型、其他 ASR、adaptive attack。

可学习点：

- “小改动、大兼容性”是非常有力的防御卖点：不改结构、不重训、低开销。
- 先做分布观察再设计方法，因果链清楚。
- 实验中 FPR/FNR 和参数选择写得很细，说明检测系统的可用性。

### IMPERio

论文类型：现实增强型攻击。

主线：

1. 现有 ASR adversarial examples 多数只能直接输入模型，或只在特定房间/特定硬件/可感知条件下 over-the-air。
2. 真实攻击必须能经过房间传播后仍让 ASR 输出目标文本。
3. 核心洞察：房间传播可以用 RIR 卷积建模；对 RIR 分布做优化，相当于音频域的 expectation over transformation。
4. 方法：采样房间大小、混响时间、speaker/microphone 位置，把 RIR convolution 作为 DNN 前置层，通过反传直接优化 raw audio；可结合 psychoacoustic hiding。
5. 实验：Kaldi hybrid ASR，不同房间、距离、混响、音频内容、无直视条件、generic vs adapted attack。
6. Discussion：generic attack 甚至可优于 adapted attack；讨论 end-to-end ASR、black-box 迁移、防御方向。

可学习点：

- 攻击论文要从“能攻击模型”升级到“能攻击真实环境”。
- 用已有领域思想迁移非常有效：把视觉 EOT 转到音频 RIR。
- 相关工作对比非常强：每类 prior attack 都指出一个现实限制，最后自然推出自己的 novelty。

### Wang: Defeating Hidden Audio Channel Attacks via Surface Vibrations

论文类型：跨模态检测防御。

主线：

1. VCS 普及，hidden voice command 能让设备执行人听不懂的命令。
2. 音频域防御容易被语音合成或特征优化伪造。
3. 核心洞察：音频播放在设备表面诱发的振动，由音频、设备物理结构、传感器采样共同决定，难以从软件音频直接伪造。
4. 方法：frontend/backend playback；motion sensor 采集振动；校准、分段、统计特征和 MFCC/chroma 特征、feature selection；supervised 和 unsupervised detection。
5. 可行性分析先于系统设计：证明 motion sensor 可捕捉语音、非线性 aliasing、振动域和音频域不同、laser vibrometer 观察差异。
6. 实验：多手机、多播放模式、Raspberry Pi prototype、13,000 traces、未知 speaker/command、partial playback、不同使用场景。
7. 结论：低成本传感器可作为独立或多模态认证机制，部署延迟可通过 partial playback 降低。

可学习点：

- 如果方法依赖一个“不直观”的物理信号，必须专门写 premise & feasibility analysis。
- 先证明信号存在，再证明信号可区分，再证明系统可部署。
- frontend/backend 两种模式把部署质疑提前化解。

### CommanderGabble

论文类型：黑盒现实攻击 + countermeasure。

主线：

1. ASR 误读并非偶然，口音、同音词、无意义声音、语速都可能造成系统性误读。
2. 现有 hidden command 攻击常依赖白盒模型、特殊硬件或大量数据。
3. 核心洞察：fast speech 会导致 phoneme reduction、replacement、coalescence；攻击者可反向构造“人听不懂但 ASR 还原成目标命令”的语音。
4. 方法：phonetic reconstruction、speech synthesis、audio transmission、winnowing/updating；失败后更新 syllabification rule。
5. Threat model：黑盒、离线生成、OTW/OTA、无需特定 speaker/mic。
6. 实验：7 个实际 ASR，100 条命令 OTW，18 条命令 OTA，家庭/会议/车载三场景，人类可理解性实验，xRef 防御。
7. Related Work：按 white-box、black-box、hardware-assisted、misinterpretation-based 分类，定位为 fast-speech misinterpretation 的首个系统化利用。

可学习点：

- 攻击论文可以用语言学机制作为核心贡献，而不是只靠模型优化。
- winnowing/updating 让攻击从“构造规则”变成“可自动搜索系统”。
- 人类实验是证明 stealthiness 的关键，不可省略。

## 可复用论文模板

### Attack 类论文模板

```text
1 Introduction
- Voice/ASR systems are widely deployed.
- Existing attacks reveal vulnerabilities, but they are limited by [white-box / special hardware / non-realistic channel / perceptibility].
- We identify [new exploitable mechanism].
- We design [attack system] that turns [mechanism] into [targeted effect].
- We evaluate on [real systems] under [real scenarios].

2 Background
- ASR pipeline or speech/phonetic/acoustic background.
- Prior attack assumptions needed for understanding.

3 Threat Model
- Attacker knowledge and capabilities.
- Attack goal and success criterion.
- OTW/OTA or black-box/white-box assumptions.

4 Attack Design
- Overview figure.
- Core mechanism.
- Algorithmic pipeline.
- Candidate generation and filtering.

5 Evaluation
- Platforms and commands.
- Attack success.
- Real-world robustness.
- Stealthiness / human study.
- Physical setup details for OTA experiments.
- Ablation and failure cases.

6 Countermeasures / Discussion
- Potential defenses.
- Transferability and limitations.

7 Related Work / Conclusion
```

### Defense/detection 类论文模板

```text
1 Introduction
- Attack is realistic and consequential.
- Existing defenses fail on [timing/deployability/assumption/usability].
- We identify [signal/domain/intermediate representation] that separates benign from malicious.
- We design [defense/detector] with [low overhead / no retraining / black-box compatibility].
- We evaluate security, utility, robustness, and efficiency.

2 Background and Related Work
- Explain attack class and system pipeline.
- Position existing defenses by limitation.

3 Problem Setting / Threat Model
- Defender and attacker capabilities.
- Desired security and utility goals.
- Formal metrics.

4 Method
- Observation and intuition.
- Formal definition.
- System architecture.
- Algorithm and parameters.

5 Evaluation
- Datasets/platforms/baselines.
- Main effectiveness.
- Utility/FPR/user study.
- Transferability/robustness.
- Adaptive attacker analysis.
- Efficiency.

6 Limitations and Future Work
- Language, noise, stronger adaptive attack, platform coverage.
- Ethical considerations if using human voices or live services.

7 Conclusion
```

## 修改论文时的检查表

### Introduction 检查

- 是否在第一页说清了现实场景和具体危害？
- 是否把 existing work 的缺口压缩成 2-3 个明确限制？
- 是否有一句可复述的核心洞察？
- 是否提前说了 threat model 的关键假设？
- 贡献列表是否按“发现 -> 系统 -> 评估 -> 含义”排序？
- 是否避免了只写“we propose a novel method”而没有解释为什么 novel？

### Method 检查

- 是否先定义攻击者/防御者能力？
- 是否明确输入、输出、成功指标？
- 是否有总览图？
- 是否从观察或物理/模型机制推出方法，而不是直接堆模块？
- 关键参数是否有选择依据？
- 如果依赖新信号，是否有 feasibility analysis？

### Evaluation 检查

- 是否每个实验都对应一个 claim？
- 是否覆盖 realistic setting，而不仅是 clean dataset？
- 如果是物理语音实验，是否报告设备、距离、音量、背景噪声、房间或 `T60`？
- 是否有强 baseline？
- Attack 是否报告 stealthiness 和 OTA？
- Defense 是否报告 FPR、utility、runtime？
- Defense 是否有 adaptive attacker analysis？
- 是否有 robustness/transferability？
- 是否主动讨论 failure cases？

### Related Work 检查

- 是否按方法类别组织，而不是按论文逐篇介绍？
- 是否有一张对比表清楚展示 threat model、assumption 和本文边界？
- 每类相关工作是否都写出“它解决什么”和“为什么不够”？
- 是否明确本文与最接近工作的区别？
- 是否避免贬低 prior work，而是强调假设差异和场景差异？

### Discussion/Limitations 检查

- 是否主动回答 adaptive attack、其他平台、其他语言、噪声、部署成本？
- 是否交代伦理、IRB、数据隐私或 responsible disclosure 等合规问题？
- 是否说明 limitation 不影响核心主张？
- 是否给出自然的 future work，而不是泛泛而谈？

## 适合你后续论文的写作取向

如果你的工作也是语音差异、ASR 攻防、音频扰动或防御方向，建议优先采用下面这条叙事：

```text
现实语音入口正在扩大
-> 现有攻击/防御忽略了某个真实差异或真实约束
-> 我们发现 [某种差异] 在 [音频域/频谱域/模型中间层/物理振动域/语言学结构] 中稳定存在
-> 该差异可以被转化为 [攻击/检测/防御] 机制
-> 在多平台、多场景、多指标下验证其安全性与可用性
```

最重要的写作原则：

- 讲问题时要真实，讲方法时要机制化，讲实验时要多场景，讲贡献时要边界清楚。
- 不要把论文写成“我提出了一个模型”。ACSAC 更喜欢“我识别了一个真实系统中的安全边界，并证明它能被利用或修补”。
- 对语音安全论文，human perception、over-the-air、low overhead、FPR、transferability 往往和主 accuracy 一样重要。
