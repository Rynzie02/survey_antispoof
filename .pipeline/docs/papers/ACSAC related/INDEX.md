# ACSAC Related Papers — Index

> **用途说明**：本文件夹收录的论文用于理解 ACSAC 的行文逻辑与写作规范，**不作为本研究的引用/参考文献**。

---

## 已解析论文（含 full.md）

### 1. Park et al. 2021 — Detecting Audio Adversarial Examples with Logit Noising
**发表**：ACSAC '21 | **机构**：POSTECH

**核心问题**：ASR 系统易受对抗音频攻击，现有防御要么精度下降要么时间开销大。

**方法**：在 ASR 解码器的 logits 层注入噪声，利用良性音频与对抗样本 logit 分布差异进行检测，无需模型结构改动或重训练。

**关键结论**：over-line 攻击检测率 100%，over-air 攻击 >95%，时间开销低于 SOTA。

**ACSAC 行文特点**：
- Abstract 直接点出现有方案的两个具体缺陷（accuracy drop / time overhead），再提出方案
- Introduction 末尾有明确的 "Our contributions are as follows" 列表（3条）
- 章节结构在 Introduction 末尾显式声明（"The remainder of this paper is organized as follows"）
- Background 单独成节（Section 2），先讲 ASR 系统三阶段再讲攻击

---

### 2. Zhang et al. 2021 — CommanderGabble: A Universal Attack Against ASR Systems Leveraging Fast Speech
**发表**：ACSAC '21 | **机构**：University of Oklahoma

**核心问题**：现有隐藏语音命令攻击依赖白盒知识或特殊硬件，通用性差。

**方法**：利用快速语音导致的音素误识别（reduction/replacement/coalescence），操控音素结构使 ASR 误识别为目标命令，无需模型知识。

**关键结论**：over-the-wire 96/100 命令成功，over-the-air 对 Alexa/Google/Cortana 成功率 93-97%。

**ACSAC 行文特点**：
- Abstract 先描述攻击场景，再点出现有方案局限，最后给出量化结果
- Introduction 用具体例子（"Open the door" → "Oh panda our"）直观展示攻击原理
- 技术挑战拆解为两个子问题，各自对应一个解决方案
- Contributions 列表在 Introduction 末尾，包含防御机制（攻防并提）

---

### 3. Wang et al. 2019 — Defeating Hidden Audio Channel Attacks on Voice Assistants via Audio-Induced Surface Vibrations
**发表**：ACSAC '19 | **机构**：Rutgers University / UAB

**核心问题**：音频域特征易被语音合成攻击绕过，需要更难伪造的检测维度。

**方法**：利用运动传感器捕获语音命令的振动签名，音频域特征与振动域特征联合检测，攻击者需同时伪造两个域。

**关键结论**：99.9% 检测准确率，支持部分语音（0.5秒）验证，可集成到现有设备。

**ACSAC 行文特点**：
- Abstract 强调 "why vibration"（核心洞察）而非直接描述方法
- Introduction 有专门的 "Why Vibration?" 小节解释设计动机
- 贡献列表分为三点：检测方案、系统设计、防御评估（对应论文三大部分）
- 实用性论证贯穿全文（低成本传感器、无需额外硬件）

---

### 4. Schönherr et al. 2020 — IMPERio: Robust Over-the-Air Adversarial Examples for Automatic Speech Recognition Systems
**发表**：ACSAC '20 | **机构**：Ruhr University Bochum

**核心问题**：现有对抗音频样本无法在真实空间（over-the-air）中保持鲁棒性，依赖特定房间参数。

**方法**：用房间冲激响应（RIR）生成器模拟多种房间环境，将 RIR 卷积作为额外网络层加入反向传播，结合心理声学掩蔽减少可感知扰动。

**关键结论**：无需先验房间知识，可在任意房间、无直视路径、数米距离下成功攻击 Kaldi ASR。

**ACSAC 行文特点**：
- Abstract 先列举现有 over-the-air 方案的三类局限，再定位本文贡献
- Introduction 有详细的 Related Work 对比段落，逐一指出前人工作的具体不足
- "Key insight" 明确标出（RIR 卷积建模），便于读者抓住核心
- 贡献列表 3 条，与论文章节一一对应

---

### 5. Liu et al. 2023 — Protecting Your Voice from Speech Synthesis Attacks
**发表**：ACSAC '23 | **机构**：Iowa State University

**核心问题**：现有语音合成防御以检测为主（事后），缺乏主动预防；Attack-VC 的预防方案存在白盒依赖、效率低、可用性差三个问题。

**方法**：在频域识别对语音合成模型影响大但人耳感知小的关键频率，修改后发布；提供 sample-level 和 speaker-level 两种方案。

**关键结论**：黑盒设置下显著降低合成质量，处理后语音仍可正常使用，用户研究验证可用性。

**ACSAC 行文特点**：
- Abstract 明确区分两种防御方案（sample-level / speaker-level），量化"extensive experiments"
- Introduction 先描述攻击危害（具体金额案例：$243K、$35M），再引出防御需求
- 对 Attack-VC 的批评非常具体（三个 limitations 逐条列出），为本文定位服务
- Background 节先讲 VC 和 TTS 框架，再讲攻击流程，最后讲现有防御

---

## 文件夹内其他论文（无 full.md，仅文件夹名）

以下论文存放于本文件夹但未解析，**同样仅供了解 ACSAC 投稿风格，不作引用**：

| 文件夹名 | 备注 |
|---------|------|
| Abbasihafshejani 等 - VocalBridge 2026 | 扩散桥净化，破解保护扰动 |
| Bai 等 - DAP 2024 | 扩散净化用于说话人验证 |
| Bai 等 - MDD 2025 | 掩码扩散检测器 |
| Bai 等 - 2024 Two-Stage Diffusion | 两阶段扩散净化 |
| Chen 等 - SpeakerGuard 2022 | 说话人识别对抗样本理解 |
| Chen 等 - 2024 Textual-Driven Purification | 文本驱动净化 |
| Fan 等 - De-AntiFake 2025 | 重新思考保护扰动 |
| Guo 等 - WavePurifier 2024 | 波形净化 |
| Haung 等 - AttackVC 2021 | 语音转换对抗攻击 |
| Hu 等 - VoiceCloak 2025 | 多维度防克隆框架 |
| Hussain 等 - WaveGuard | 波形防护 |
| Jin 等 - Whispering Under The Eaves | — |
| Kang 等 - DiffAttack 2024 | 扩散净化逃逸攻击 |
| Kassis 等 - DiffBreak 2025 | 扩散净化鲁棒性分析 |
| Kong 等 - DiffWave 2021 | 扩散模型音频合成 |
| Li 等 - CloneShield 2025 | 零样本语音克隆通用扰动 |
| Li 等 - Voice Guard 2023 | 时域对抗扰动保护隐私 |
| Mincheol Park 等 - Adversarial Purification via SR | 超分辨率+扩散净化 |
| Tan 等 - DualPure 2024 | 语音命令识别双重净化 |
| Wu 等 - AudioPure 2023 | 扩散模型防御对抗音频 |
| Yu 等 - AntiFake 2023 | 对抗音频防止未授权语音合成 |
| Zhang 等 - E2E-VGuard 2025 | LLM 端到端语音合成防御 |
| Zhang 等 - SafeSpeech 2025 | 鲁棒通用语音保护 |

---

## ACSAC 行文逻辑总结

基于以上5篇论文归纳的 ACSAC 写作规范：

**结构模式**：
1. **Abstract**：问题 → 现有方案局限（具体化）→ 本文方法（1-2句）→ 量化结果
2. **Introduction**：背景+动机 → 具体攻击/防御场景（含真实案例或数字）→ 现有工作不足（逐条）→ 本文方案核心洞察 → Contributions 列表（3-4条）→ 章节结构声明
3. **Background**：先讲系统架构，再讲攻击模型，最后讲现有防御（为 Related Work 铺垫）
4. **Related Work**：通常嵌入 Introduction 或单独成节，逐一指出前人局限

**写作风格**：
- Contributions 必须是可验证的具体声明，不用模糊动词（"we explore" → "we propose/demonstrate/show"）
- 量化结果在 Abstract 中就出现（成功率、准确率、开销数字）
- 攻击类论文需包含防御讨论；防御类论文需包含对攻击者适应性的分析
- 安全论文强调 threat model 的明确性（白盒/黑盒、over-line/over-air）
