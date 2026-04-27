# VoiceCloak: A Multi-Dimensional Defense Framework against Unauthorized Diffusion-based Voice Cloning

Qianyue $\mathbf { H } \mathbf { u } ^ { 1 }$ , Junyan $\mathbf { W } \mathbf { u } ^ { 1 }$ , Wei $\mathbf { L u } ^ { 1 * }$ , Xiangyang Luo 2

1School of Computer Science and Engineering, MoE Key Laboratory of Information Technology, Guangdong Province Key Laboratory of Information Security Technology, Sun Yat-sen University, Guangzhou 510006, China 2State Key Laboratory of Mathematical Engineering and Advanced Computing, Zhengzhou, 450002, China huqy $5 6 @$ mail2.sysu.edu.cn, wujy $2 9 8 @$ mail2.sysu.edu.cn, luwei3 $@$ mail.sysu.edu.cn, luoxy ieu@sina.com

# Abstract

Diffusion Models (DMs) have achieved remarkable success in realistic voice cloning (VC), while they also increase the risk of malicious misuse. Existing proactive defenses designed for traditional VC models aim to disrupt the forgery process, but they have been proven incompatible with DMs due to the intricate generative mechanisms of diffusion. To bridge this gap, we introduce VoiceCloak, a multi-dimensional proactive defense framework with the goal of obfuscating speaker identity and degrading perceptual quality in potential unauthorized VC. To achieve these goals, we conduct a focused analysis to identify specific vulnerabilities within DMs, allowing VoiceCloak to disrupt the cloning process by introducing adversarial perturbations into the reference audio. Specifically, to obfuscate speaker identity, VoiceCloak first targets speaker identity by distorting representation learning embeddings to maximize identity variation, which is guided by auditory perception principles. Additionally, VoiceCloak disrupts crucial conditional guidance processes, particularly attention context, thereby preventing the alignment of vocal characteristics that are essential for achieving convincing cloning. Then, to address the second objective, VoiceCloak introduces score magnitude amplification to actively steer the reverse trajectory away from the generation of high-quality speech. Noise-guided semantic corruption is further employed to disrupt structural speech semantics captured by DMs, degrading output quality. Extensive experiments highlight VoiceCloak’s outstanding defense success rate against unauthorized diffusion-based voice cloning. Audio samples of VoiceCloak are available at https://voice-cloak.github.io/VoiceCloak/.

# 1 Introduction

Diffusion Models (DMs) Ho, Jain, and Abbeel (2020); Song, Meng, and Ermon (2020); Rombach et al. (2022) have recently emerged as powerful generative tools, achieving unprecedented success within realistic voice cloning (VC). Their iterative denoising process enables generating speech with remarkable naturalness, detail, and fidelity to human voice Popov et al. (2022); Kong et al. (2020); Jeong et al. (2021); Shen et al. (2024). However, the open-source availability and ease of use of these models intensify concerns about potential misuse. Attackers can synthesize highly realistic voice replicas from short public audio clips, as depicted in Figure 1(a), enabling sophisticated fraud and circumvention of voiceprint authentication.

![](images/b8542de5041ffe8e368fe5c3728bb357be6ecc1697ff32d419af24665e10f55e.jpg)  
Figure 1: Illustration of diffusion-based voice cloning malicious misuse. (a) Voice forgery enables threats of fraud. (b) Traditional methods struggle due to ineffective disruptive gradients. (c) Audio protected by VoiceCloak resists highfidelity cloning.

To counter such unauthorized use, two main defense paradigms are introduced, including forgery detection and proactive disruption. Reactive detection methods Jung et al. (2022); Zhou and Lim (2021); Wu et al. (2024) identify forgeries after they are crafted, often too late to prevent harm. This highlights the need for proactive defenses that disrupt the synthesis process itself. Prior proactive work Huang et al. (2021); Yu, Zhai, and Zhang (2023); Chen et al. (2024); Li et al. (2023) has focused on adding imperceptible adversarial perturbations to reference audio by compromising the functionality of either the voice decoder or the speaker identity encoder.

However, existing defenses designed for prior architectures are largely ineffective against Diffusion Models (DMs). This incompatibility arises from two fundamental challenges: gradient vanishing and dynamic conditioning (Figure 1). Specifically, (1) the single forward pass gradient computation relied upon by many defenses become unreliable or vanish within the multi-step denoising process of DMs and the corresponding deep computational graph, rendering such single-pass gradient information ineffective for disrupting the full generation trajectory Kang, Song, and Li (2023). (2) Strategies targeting specific subnetworks (e.g. speaker or content encoders) fail because DMs often employ dynamic conditioning mechanisms, which means no single modules solely responsible for condition processing. Consequently, methods targeting individual components struggle to cause global disruption. These fundamental incompatibilities underscore the need for novel strategies tailored to this generative paradigm.

In light of the incompatibility of prior defenses with the diffusion paradigm, we introduce VoiceCloak, a novel proactive defense framework designed for two primary objectives against unauthorized voice cloning: Speaker Identity Obfuscation and Perceptual Fidelity Degradation. Driven by these two objectives, we conduct an analysis to identify and exploit corresponding intrinsic vulnerabilities within Diffusion Models (DMs). Based on this analysis, we design specific optimization objectives for the protective perturbation to effectively disrupt the synthesis process.

To achieve identity obfuscation, VoiceCloak first directly manipulates speaker representations within a universal embedding space, guided by psychoacoustic principles to maximize perceived identity distance and hinder the DM’s identity signature extraction. Second, recognizing that convincing mimicry depends on attention mechanisms in conditional guidance to align speaker style with content, we exploit this by introducing attention context divergence. This design prevents the attention mechanism from correctly utilizing contextual information, thereby disrupting the alignment required for accurate cloning.

Simultaneously, to achieve fidelity degradation, we focus on vulnerabilities within the core generative process itself. First, we employ Score Magnitude Amplification (SMA) to exploit the sensitivity of the iterative denoising trajectory which is crucial for realistic output, steering the generation path away from high-fidelity regions. Furthermore, acknowledging that the U-Net’s understanding of highlevel semantics governs output naturalness, we utilize noiseguided semantic corruption to disrupt the capture of structural features to promote incoherence within the noise semantic space and degrade generation quality. These goaldriven strategies which originates from adversarial analysis form a comprehensive defense. Extensive experiments confirm the superior defense efficacy against diffusion-based VC attacks under equivalent perturbation budgets.

Our contributions are summarized as follows:

• We propose VoiceCloak, a novel defense framework against Diffusion-Based VC that prevents unauthorized voice ”theft” by exploiting intrinsic diffusion vulnerabilities through multi-dimensional adversarial interventions. • We introduce auditory-perception-guided adversarial perturbations into speaker identity representations and disrupt the diffusion conditional guidance process to effectively distort identity information in synthesized audio. • We present SMA, which controls the score function to divert the denoising trajectory, complemented by a semantic function designed to adversarially corrupt structural semantic features within the U-Net, degrading the perceptual fidelity of the forged audio.

# 2 Related Work

# 2.1 Audio Diffusion Models

Diffusion Models Ruan et al. (2023); Kawar et al. (2023); Ruiz et al. (2023) a dominant force in generative modeling, demonstrating extraordinary performance across multimodal tasks, significantly advancing audio synthesis tasks like text-to-speech Popov et al. (2021); Jeong et al. (2021) and unconditional audio generation Kong et al. (2020); Liu et al. (2023). Particularly within voice cloning (VC), diffusion-based methodsPopov et al. (2022); Choi, Lee, and Lee (2024), mostly leveraging score-based formulations via stochastic differential equations (SDEs) Song et al. (2020), now yield outputs with remarkable naturalness and speaker fidelity. While impressive, this state-of-the-art performance significantly heightens concerns regarding potential misuse, directly motivating proactive defense strategies such as the framework proposed herein.

# 2.2 Proactive Defense via Adversarial Examples

Beyond passive DeepFake detection, proactive defenses aim to preemptively disrupt malicious syntheses, by introducing adversarial perturbations to the original audio. Early work Huang et al. (2021) demonstrated the feasibility of this approach but struggled to balance effectiveness with imperceptibility. Subsequent research focused on improving this trade-off. Strategies included using psychoacoustic models to enhance imperceptibility Li et al. (2023), incorporating human-in-the-loop refinement for better balance Yu, Zhai, and Zhang (2023), , and employing GAN-based generators to improve efficiency Dong et al. (2024).

Despite these advancements, prior proactive strategies were largely designed for earlier generative architectures like GANs. They often overlook the unique mechanisms and internal structures of Diffusion Models (DMs), limiting their applicability. Recognizing this critical gap, our work proposes a defense specifically tailored to the challenges of diffusion-based voice cloning.

# 3 Preliminaries

# 3.1 Score-based Diffusion

Score-based generative models define a continuous-time diffusion process using stochastic differential equations (SDEs) Song et al. (2020). The forward process gradually perturbs clean data $\mathbf { x } _ { 0 } \sim p _ { 0 } ( \mathbf { x } )$ into noise via the SDE:

$$
\begin{array} { r } { d \mathbf { x } = f ( \mathbf { x } , t ) d t + g ( t ) d \mathbf { w } , } \end{array}
$$

where $f ( \mathbf { x } , t )$ and $g ( t )$ are the drift and diffusion coefficients, and w is a Wiener process. The corresponding reverse-time SDE that transforms $x _ { T }$ back into $p _ { 0 } ( \mathbf { x } )$ can be expressed as:

$$
\begin{array} { r } { d \mathbf { x } = \left[ f ( \mathbf { x } , t ) - g ( t ) ^ { 2 } \nabla _ { \mathbf { x } } \log p _ { t } ( \mathbf { x } ) \right] d t + g ( t ) d \bar { \mathbf { w } } , } \end{array}
$$

where $\bar { \bf w }$ is the reverse-time Wiener process and $\nabla _ { \mathbf { x } } \log p _ { t } ( \mathbf { x } )$ represents the score-function. Then, the

score-based diffusion model is trained to estimate the score function:

$$
\arg \operatorname* { m i n } _ { \theta } \lambda _ { t } \mathbb { E } _ { p _ { t } ( x ) } \left\| s _ { \theta } ( \mathbf { x } _ { t } , t ) - \nabla \log p _ { t } ( \mathbf { x } _ { t } | \mathbf { x } _ { 0 } ) \right\| _ { 2 } ^ { 2 } ,
$$

where the expectation is taken over the data distribution $p _ { 0 } ( \mathbf { x } _ { 0 } )$ and the transition kernel $p _ { t } ( \mathbf { x } _ { t } | \mathbf { x } _ { 0 } )$ .

# 3.2 Adversarial Vulnerability Analysis

As mentioned before, the optimization of perturbation is guided by two core objectives: (1) Speaker Identity Obfuscation and (2) Perceptual Fidelity Degradation. The design of these perturbations stems from a targeted analysis to identify and exploit specific vulnerabilities within the Diffusion Model (DM) generative process itself.

A primary objective of VoiceCloak is speaker identity obfuscation. Convincing identity mimicry in Diffusion Models (DMs) depends on precise conditional control which is guided by acoustic details modeled from the reference audio $x _ { r e f }$ . This guidance critically relies on mechanisms that align the target speaker’s acoustic characteristics from $x _ { r e f }$ with the phonetic content from $x _ { s r c }$ . Specifically, the attention block is responsible for executing this alignment. We identify this crucial acoustic style-to-content mapping as a key vulnerability,as inaccurate alignment directly compromises the successful rendering of the target speaker’s identity. Consequently, disrupting this attention-driven pathway offers a strategy for identity obfuscation.

Complementary to identity obfuscation, the second objective is perceptual fidelity degradation, aimed at diminishing the usability of any synthesized speech. This involves targeting the core denoising process of DMs. The high fidelity of Diffusion-based VC relies on the model learning a precise reverse denoising trajectory to progressively refine noisy sample $x _ { t }$ towards the natural audio distribution. This reliance on trajectory precision presents an exploitable vulnerability. Therefore, adversarially diverting the trajectory can disrupt convergence towards the high quality audio region on the manifold.

Additionally, we target the U-Net’s internal feature representations as a vulnerability for degrading perceptual quality. Prior attribute editing work Ronneberger, Fischer, and Brox (2015); Oh, Lee, and Lee (2024); Tumanyan et al. (2023) confirms that these hierarchical features within the U-Net are controllable and also encode crucial semantic and acoustic details that govern the coherence and naturalness of the synthesized speech. Therefore, adversarially corrupting these representations directly impair the model’s ability to synthesize natural-sounding speech, offering a complementary strategy for fidelity degradation. This analysis reveals critical vulnerabilities in DMs, providing targeted avenues for adversarial intervention.

# 4 Methodology

To provide the necessary background, the problem formulation is firstly established in Section 4.1. As shown in Figure 2, the VoiceCloak consists of four sub-modules, which will be introduced separately next.

# 4.1 Problem Formulation

Assume that malicious users can obtain reference audio $x _ { r e f }$ of a target speaker. Leveraging open-source voice cloning models, they can synthesize speech which mimics the vocal characteristics of $x _ { r e f }$ , denoted $\mathcal { V } \mathcal { C } ( x _ { s r c } , x _ { r e f } , t )$ . Our goal is to proactively safeguard $x _ { r e f }$ against unauthorized voice cloning. We achieve this by introducing an imperceptible adversarial perturbation $\delta$ to create a protected version $x _ { a d v } = x _ { r e f } + \delta$ . This perturbation is optimized to disrupt the diffusion synthesis process when conditioned on $x _ { a d v }$ . Formally, we aim to find an optimal $\delta$ that maximizes the dissimilarity between the outputs generated using $x _ { r e f }$ and $x _ { a d v }$ .

$$
\begin{array} { r l } & { \arg \operatorname* { m a x } _ { \delta } \mathcal { D } ( \mathcal { V C } ( { x } _ { s r c } , { x } _ { r e f } , t ) , \mathcal { V C } ( { x } _ { s r c } , { x } _ { a d v } , t ) ) , } \\ & { \mathrm { s u b j e c t \ t o } \quad \| \delta \| _ { \infty } \leq \epsilon , } \end{array}
$$

where $\mathcal { D } ( \cdot )$ measures the output discrepancy, $t$ is the diffusion timestep, and $\epsilon$ is the $l _ { \infty }$ -norm budget for the perturbation $\delta$ . Our approach focuses on designing specific adversarial objectives that implicitly define $\mathcal { D }$ by exploiting intrinsic vulnerabilities within the diffusion mechanism itself.

# 4.2 Adversarial Identity Obfuscation

Opposite-Gender Embedding Centroid Guidance Inspired by prior work showing dedicated speaker embeddings capture identity Chen et al. (2024), directly manipulating these embeddings offers a direct approach, but its practical utility is hindered by poor transferability when attacking unknown or different encoder models.

Therefore, we explore leveraging speech Representation Learning Chen et al. (2022b) to extract general acoustic representations that inherently capture speaker identity cues. Specifically, we select WavLM Chen et al. (2022a) as our representation extractor, denoted as $\mathcal { R } ( \cdot )$ . By applying perturbations within this more general feature space, we aim for broader effectiveness against various models. As a baseline untargeted objective, we consider maximizing the embedding distance:

$$
\mathcal { L } _ { I D } = 1 - S i m ( \mathcal { R } ( x _ { a d v } ) , \mathcal { R } ( x _ { r e f } ) ) ,
$$

where $S i m ( x _ { 1 } , x _ { 2 } )$ represents the cosine similarity metric. While this untargeted objective effectively pushes $x _ { a d v }$ away from the reference identity representation domain, it lacks specific directionality in high-dimensional space. So we further incorporate a targeted component guided by psychoacoustic principles.

Psychoacoustic studies Kreiman and Sidtis (2011); Lavner, Gath, and Rosenhouse (2000) indicate that significant perceptual differences in speaker identity often exist between genders, linked to specific acoustic cues like F0 and formant structures . Leveraging this, we assume that guiding the adversarial embedding towards the opposite gender one will likely create the strongest perceptual contrast, enhancing identity obfuscation.

Based on this insight, we propose an auditory-perceptionguided adversarial perturbation. Specifically, we first randomly select a speaker from the dataset whose gender is opposite to that of $x _ { r e f }$ . To establish a representative identity embedding, we compute the centroid $\mathcal { C } _ { o p p }$ by averaging the embeddings of all utterances in $\mathcal { X }$ within the WavLM feature space $\mathcal { R }$ :

![](images/d48e515532cd7f26d14b708e18ccde7ab1b544d6c89a1620d3948b7183c39b8c.jpg)  
Figure 2: Overview of the proposed framework. Perturbation optimization is guided by gradients from $\mathcal { L } _ { t o t a l }$ , aggregating four targeting two objectives: (1) Identity Obfuscation (via Opposite-Gender Centroid Guidance and Attention Context Divergence) and (2) Perceptual Fidelity Degradation (Score Magnitude Amplification and Noise-Guided Semantic Corruption).

$$
\mathcal { C } _ { o p p } = \frac { 1 } { N } \sum _ { x _ { i } \in \mathcal { X } } \mathcal { R } ( x _ { i } ) ,
$$

$$
\mathcal { L } _ { I D } = - S i m ( \mathcal { R } _ { a d v } , \mathcal { R } _ { r e f } ) + \underbrace { S i m ( \mathcal { R } _ { a d v } , \mathcal { C } _ { o p p } ) } _ { \mathrm { G e n d e r } } ,
$$

where $\mathcal { X }$ is the set of opposite gender utterances, $N$ represents length of $\mathcal { X }$ , $\mathcal { R } _ { i }$ is the embedding of $x _ { i }$ in space $\mathcal { R }$ .

Finally, minimizing the Eq. 7 directs the optimization process to simultaneously dissociate from the original speaker identity and converge towards a region selected based on psychoacoustic principles to maximize identity ambiguity.

Attention Context Divergence Building upon the motivation to disrupt conditional guidance for identity obfuscation (Section 3.2), we introduce Attention Context Divergence. This strategy targets the attention mechanism to interfere with its use of contextual information from $x _ { r e f }$ .

Commonly in diffusion-based VC, conditional information from $x _ { r e f }$ is integrated via Linear-attention layers Katharopoulos et al. (2020). Within the U-Net, latent code representing the content $x _ { s r c } ^ { t }$ are linearly projected to $Q$ matrix, whereas condition latents $\boldsymbol { x } _ { r e f } ^ { t }$ are projected to $K$ and

$V$ . Linear-attention first computes a context matrix by aggregating values $( V )$ weighted by keys $( K )$ :

$$
\mathcal { H } _ { c t x } ( x _ { r e f } ) = \mathrm { S o f t m a x } ( \phi ^ { ( l , t ) } ( x _ { r e f } ^ { t } ) W _ { K } ^ { l } ) ( \phi ^ { ( l , t ) } ( x _ { r e f } ^ { t } ) W _ { V } ^ { l } ) ^ { T } ,
$$

where $\mathcal { H } _ { c t x }$ represents the context hidden state, $\phi ^ { ( l , t ) } ( \cdot )$ is the deep features of the $l ^ { t h }$ block in U-Net at timestep $t$ , and $W _ { K } ^ { i } , W _ { V } ^ { l }$ are the projection matrices. Then, the context interacts with queries $( Q )$ to obtain the final attention output: $\mathcal { A } ^ { l } = ( \mathcal { H } _ { c t x } ) ^ { T } ( W _ { Q } ^ { l } \phi ^ { ( l , t ) } ( x _ { s r c } ^ { t } ) )$ . This context representation provides a dynamic summary of the reference speaker’s stylistic features, weighted by their relevance to the current content queries during synthesis. By applying a softmax function, we obtain an explicit probability distribution over sequence positions which signifies the model’s information focus. Our strategy, therefore, is to maximize the Kullback-Leibler (KL) divergence Kullback and Leibler (1951) between the context distribution derived from the original reference and the adversarial audio:

$$
\begin{array} { r l } & { \mathcal { L } _ { c t x } = D _ { K L } ( P _ { r e f } \parallel P _ { a d v } ) , } \\ & { P _ { a d v } = \mathrm { S o f t m a x } ( \mathcal { H } _ { c t x } ( x _ { a d v } ) ) , } \end{array}
$$

Maximizing this divergence forces the attention pattern to deviate from the original, thereby hindering accurate style alignment.

To enhance the impact on identity, we directs the adversarial pressure on the U-Net’s downsampling path. The rationale is that these earlier layers primarily process coarser, lower-frequency features Wang et al. (2024) strongly associated with speaker timbre and identity. Therefore, focusing our proposed loss $\mathcal { L } _ { c t x }$ on the attention layers within the U-Net’s downsampling path, the calculation of $P _ { a d v }$ can be restated as:

$$
P _ { a d v } = \mathrm { S o f t m a x } ( \sum _ { l } \mathcal { H } _ { c t x } ^ { l } ( x _ { a d v } ) ) ,
$$

where the layer index $l$ iterates over the set Down $( l \in$ Down), representing the U-Net’s downsampling blocks.

# 4.3 Perceptual Fidelity Degradation

Score Magnitude Amplification To degrade perceptual fidelity by exploiting the sensitivity of the denoising trajectory, we introduce Score Magnitude Amplification. This design directly interferes with the score function $s _ { \theta }$ , which is estimated by the U-Net according to Eq. 3 and provides the essential drift term for the reverse SDE. We posit that the magnitude of $s _ { \theta }$ relates to the strength of the drift guiding the noisy sample toward the target data manifold. Exploiting this connection, the SMA objective involves maximizing the magnitude of the score prediction $s _ { \theta }$ :

$$
\mathcal { L } _ { s c o r e } = \mathbb { E } _ { p _ { t } ( x ) , t \sim \mathcal { U } ( 1 , T _ { a d v } ) } [ | | s _ { \theta } ( x _ { s r c } ^ { t } , x _ { a d v } ^ { t } , t ) | | _ { 2 } ] ,
$$

where $p _ { t } ( x )$ is the distribution of noisy samples $x ^ { t } \sim$ $q ( x ^ { t } | x ^ { 0 } ) , \operatorname { \mathbb { E } } [ \cdot ]$ calculates the average value, $T _ { a d v }$ stands for adversarial timesteps which will be discussed in Section 4.4. Optimizing the above formula introduces an erroneous drift strength. Consequently, the denoising trajectory is forcefully diverted, resulting in a collapse in perceptual quality.

Furthermore, the iterative nature of the diffusion process may amplify these induced trajectory deviations. Perturbations introduced at earlier timesteps can propagate through subsequent steps. This error cumulative effect thus enhances the efficacy of our adversarial strategy.

Noise-Guided Semantic Corruption Following the motivation outlined in Section 3.2, we introduce a bidirectional semantic interference strategy. The core idea is twofold: (1) compel the features generated with $x _ { a d v }$ to diverge from those generated using the original reference $x _ { r e f }$ , and (2) concurrently guide these adversarial features towards a ”semantic-free” state.

Specifically, consider a network layer $l$ within the frozen U-Net $\mathcal { U } _ { \theta }$ and timestep $t$ , we extract the original features $f ^ { ( l , t ) } = \mathcal { U } _ { \theta } ^ { l } ( x _ { s r c } , x _ { r e f } , t )$ conditioned on $x _ { r e f }$ , and $f _ { a d v } ^ { ( l , t ) } =$ $\mathcal { U } _ { \theta } ^ { l } ( x _ { s r c } , x _ { a d v } , t )$ corresponding to the adversarial version. Furthermore, to define a ”semantic-free” target, we leverage the U-Net’s activation modes to unstructured information. We extract features $f _ { n o i s e } ^ { ( l , t ) } = \mathcal { U } _ { \theta } ^ { l } ( x _ { n o i s e } , x _ { n o i s e } , t )$ by feeding Gaussian white noise $x _ { n o i s e }$ as both the source content and the reference condition. $f _ { n o i s e } ^ { ( l , t ) }$ can be considered to represent unstructured features and lack semantic information. The bidirectional objective aims to maximize the distance between $f _ { a d v } ^ { ( l , t ) }$ and $f ^ { ( l , t ) }$ while minimizing it with $f _ { n o i s e } ^ { ( l , t ) }$ . This objective encourages the adversarial features to abandon the original semantic structure and move towards a state of incoherence which can be formalized as:

$$
\begin{array} { r } { \mathcal { L } _ { s e m } = 1 - \cos ( f _ { a d v } ^ { ( l , t ) } , f ^ { ( l , t ) } ) + \underbrace { \cos ( f _ { a d v } ^ { ( l , t ) } , f _ { n o i s e } ^ { ( l , t ) } ) } _ { \mathrm { S e m - f r e e } } , } \end{array}
$$

where we employ the cosine distance metric $\cos ( \cdot )$ as it emphasizes the structural similarity between high-dimensional features rather than their absolute error. For enhanced impact on perceptual quality, the $\mathcal { L } _ { s e m }$ is strategically applied to layers within the U-Net’s upsampling path. These layers are critical for reconstructing the fine-grained acoustic details that govern output naturalness and perceived quality.

# 4.4 Joint Optimization of Defense Objectives

The final adversarial perturbation $\delta$ for VoiceCloak is optimized to simultaneously achieve our dual objectives (Section 4.2 and 4.3). The joint objectives of this comprehensive defense are formalized as follows

$$
\begin{array} { r l } & { \mathcal { L } _ { t o t a l } = ( \mathcal { L } _ { I D } , \mathcal { L } _ { c t x } , \mathcal { L } _ { s c o r e } , \mathcal { L } _ { s e m } ) \Lambda ^ { T } , } \\ & { \qquad \delta : = \arg \operatorname* { m a x } _ { \delta } \mathcal { L } _ { t o t a l } , } \end{array}
$$

where $\Lambda = ( \lambda _ { I D } , \lambda _ { c t x } , \lambda _ { s c o r e } , \lambda _ { s e m } )$ controls the weight factors that balance the relative importance of these defenses.

The efficacy of our perturbation optimization is influenced by the choice of diffusion timesteps $T _ { a d v }$ used for gradient computation. Informed by prior work $\mathrm { Y u }$ et al. (2024) indicating early denoising steps primarily reconstruct lowfrequency overall structural signal, we concentrate the optimization on these initial steps to maximize the disruption of fundamental integrity and reduce computational overhead.

# 5 Experiments

# 5.1 Experimental Setup

Datasets Experiments are conducted on the LibriTTS Zen et al. (2019) and VCTK Yamagishi, Veaux, and MacDonald (2019) datasets. We selected a gender-balanced audio subsets (479 utterances from LibriTTS, 500 from VCTK) to generate adversarial reference speech set $\mathcal { D } _ { x _ { a d v } }$ .

Baseline Methods We compare VoiceCloak against existing voice protection methods. We adopt the following methods for fair comparison: Attack-VC Huang et al. (2021), VoicePrivacy Chen et al. (2024), and VoiceGuard Li et al. (2023). We also include a naive baseline: adding random Gaussian noise to $x _ { r e f }$ .

Evaluation Metrics For identity protection, we report the Automatic Speaker Verification (ASV) acceptance rate, where a lower rate for protected outputs signifies more effective obfuscation. We define a comprehensive Defense Success Rate (DSR) to measure the achievement of both our objectives. A defense is considered successful if the protected output both fails speaker verification and exhibits low perceptual quality $\mathrm { ( D \bar { S } } = s _ { A S V } < \tau _ { A S V } \land \mathrm { N I S Q A } ( y _ { a d v } ) <$ $\tau _ { q , \vec { \mathbf { \nabla } } }$ , with thresholds $\tau _ { A S V } = 0 . 2 5$ and $\tau _ { q } = 3 . 0$ . Additional metrics include Dynamic Time Warping (DTW) Sakoe and Chiba (1978) and SSIM between $y$ and $y _ { a d v }$ spectrograms. Perturbation imperceptibility on $x _ { a d v }$ versus $x _ { r e f }$ is measured using PESQ Rix et al. (2001), Mel-Cepstral Distortion (MCD) Kubichek (1993) and SNR.

<table><tr><td rowspan="2">Datasets</td><td rowspan="2">Methods</td><td colspan="4">Defense Effectiveness</td><td colspan="4">Imperceptibility</td></tr><tr><td>DTW↑</td><td>ASV↓</td><td>SSIM↓</td><td>NISQA↓</td><td>DSR↑</td><td>PESQ↑</td><td>MCD↓</td><td>SNR个</td></tr><tr><td rowspan="6">LibriTTS</td><td>Undefended</td><td>=</td><td>76.49%</td><td>1</td><td>3.96</td><td>-</td><td></td><td>-</td><td>1</td></tr><tr><td>Random Noise loo</td><td>2.01</td><td>55.20%</td><td>0.31</td><td>3.72</td><td>16.00%</td><td>3.37</td><td>1.35</td><td>34.80</td></tr><tr><td>Attack-VC</td><td>2.29</td><td>36.20%</td><td>0.31</td><td>3.57</td><td>30.40%</td><td>2.31</td><td>3.71</td><td>5.29</td></tr><tr><td>VoicePrivacy</td><td>2.26</td><td>20.80%</td><td>0.30</td><td>3.60</td><td>26.80%</td><td>2.99</td><td>1.37</td><td>33.25</td></tr><tr><td>VoiceGuard</td><td>2.08</td><td>16.49%</td><td>0.29</td><td>3.63</td><td>43.45%</td><td>2.15</td><td>4.39</td><td>10.58</td></tr><tr><td>Ours</td><td>2.12</td><td>11.40%</td><td>0.27</td><td>2.36</td><td>71.40%</td><td>3.22</td><td>1.29</td><td>33.53</td></tr><tr><td rowspan="6">VCTK</td><td>Undefended</td><td>-</td><td>63.68%</td><td>-</td><td>3.41</td><td>-</td><td>-</td><td>-</td><td>=</td></tr><tr><td>Random Noise loo</td><td>1.68</td><td>58.00%</td><td>0.35</td><td>3.16</td><td>11.38%</td><td>3.25</td><td>1.38</td><td>34.10</td></tr><tr><td>Attack-VC</td><td>2.05</td><td>38.50%</td><td>0.33</td><td>2.82</td><td>26.20%</td><td>2.25</td><td>3.80</td><td>4.50</td></tr><tr><td>VoicePrivacy</td><td>1.88</td><td>30.28%</td><td>0.35</td><td>2.77</td><td>39.06%</td><td>2.87</td><td>1.53</td><td>31.87</td></tr><tr><td>VoiceGuard</td><td>1.87</td><td>31.42%</td><td>0.32</td><td>3.02</td><td>22.11%</td><td>2.05</td><td>4.57</td><td>10.00</td></tr><tr><td> Ours</td><td>1.93</td><td>19.74%</td><td>0.29</td><td>2.51</td><td>63.41%</td><td>3.09</td><td>1.33</td><td>32.41</td></tr></table>

Table 1: Comparison of defense effectiveness and adversarial imperceptibility with SOTA methods. Higher values are better for metrics marked with $\uparrow$ , and vice versa for those marked with ↓. The best result is marked in BOLD.

Implementation Details We conduct our experiments mainly using DiffVC Popov et al. (2022) as the target system. We set the number of optimization iterations to 50 and a step size $\alpha = 4 \times 1 0 ^ { - 5 }$ . And $\delta$ is constrained within an $l _ { \infty }$ -norm ball of $\epsilon = 0 . 0 0 2$ . We set the adversarial and inference timesteps respectively $T _ { a d v } = 6$ and $T = 1 0 0$ . The loss function combined identity and quality objectives with weights $\Lambda = ( 1 . 0 , 4 . 5 , 1 0 , 0 . 8 \dot { 5 } )$ . All experiments are conducted on NVIDIA RTX 3090 GPU with a fixed random seed.

# 5.2 Comparison and Analysis

Comparison with Baselines As shown in Table 1, Voice-Cloak demonstrates exceptional defense efficacy, significantly outperforming all baselines with a Defense Success Rate (DSR) of over $71 \%$ on LibriTTS and $63 \%$ on VCTK. This high DSR reflects success in both of our defense objectives: identity obfuscation is achieved by drastically reducing speaker verification acceptance rates to $11 \%$ , while perceptual quality degradation is confirmed by low NISQA scores indicating unacceptable audio generation quality. This dual ability to effectively cripple both identity mimicry and audio usability distinguishes VoiceCloak from defenses that may struggle with diffusion models or focus primarily on one objective. As expected, the naive Random Noise baseline confirms that unstructured noise provides very limited protection. Regarding imperceptibility, VoiceCloak performs comparably to baselines. This supports our strategy of targeting intrinsic vulnerabilities, rather than simply increasing perturbation magnitude.

Figure 3 further visualizes disruption result. Applying VoiceCloak protection results markedly different from the undefended one. The $F _ { 0 }$ curve is notably degraded, appearing blurred and unpredictable, accompanied by highly inconsistent intonation changes.

Protecting Commercial Systems We also evaluated the effectiveness in protecting commercial speaker verification (SV) APIs (Iflytek, Azure) to simulate real-world antispoofing scenarios. Successful protection aims to minimize the similarity score returned by the API. Figure 4 demonstrated VoiceCloak’s superior ability to decouple the protected audio from the original speaker’s identity.

![](images/3cc358768c62af6817de2d351a977847ce23af01d576cffa1b9143ce8f15a5bb.jpg)  
Figure 3: Mel spectrograms with $F _ { 0 }$ pitch contours (green lines), and inferred intonation of the corresponding words. Arrows indicate perceived intonation shifts. (Intonation aligns with the ground truth, which is marked by green arrows, and diverges, which is marked by red arrows.)

User Study We conducted a user study with 50 participants to assess perceptual impact. Listeners performed comparisons on two criteria: Timbre Dissimilarity and Naturalness Disruption and we aggregated results in Figure 5, where ”Neutral” indicates no perceived difference. Participants consistently rated VoiceCloak’s outputs as having both greater timbre dissimilarity and more severe naturalness disruption, confirming its human-perceived effectiveness.

![](images/8e583d2cb84c24d629dce52761604cd9f2d8f10bec31a140d09723461359698c.jpg)  
Figure 4: Protecting commercial speaker verification APIs (Iflytek, Azure) from spoofing attacks (lower are better).

![](images/3fea09e267235dcaef75e942c492cda0283a8f374ab8790b683b9a74d5fe2947.jpg)  
Figure 5: User perceptual study results. (a) Timbre Dissimilarity Preference. (b) Corresponding results for perceived Naturalness Disruption.

Table 2: Ablation study on the contribution of different settings for Identity Obfuscation. $\checkmark$ indicates the setting is used, ”w/o” denotes the exclusion of the specified term.   

<table><tr><td colspan="2">Setings</td><td colspan="5">Defense Effectiveness</td></tr><tr><td>LID</td><td>Lctx</td><td>DTW个</td><td>ASV↓</td><td>SSIM↓</td><td>NISQA↓</td><td>DSR↑</td></tr><tr><td>1</td><td>1</td><td>1.96</td><td>46.82%</td><td>0.30</td><td>3.66</td><td>22.58%</td></tr><tr><td>√</td><td>1</td><td>2.16</td><td>8.57%</td><td>0.30</td><td>3.57</td><td>27.74%</td></tr><tr><td>w/o Gender</td><td>-</td><td>2.25</td><td>19.92%</td><td>0.30</td><td>3.60</td><td>14.40%</td></tr><tr><td>1</td><td>√</td><td>2.31</td><td>16.20%</td><td>0.27</td><td>2.96</td><td>62.57%</td></tr><tr><td>√</td><td>√</td><td>2.13</td><td>11.00%</td><td>0.27</td><td>2.85</td><td>69.20%</td></tr></table>

Table 3: Ablation study for Perceptual Fidelity Degradation. Checkmark $( \checkmark )$ indicates the setting is used,”w/o” denotes the exclusion of the specified loss term.   

<table><tr><td colspan="2">Settings</td><td colspan="5">Defense Effectiveness</td></tr><tr><td>Lscore</td><td>Lsem</td><td>DTW↑</td><td>ASV↓</td><td>SSIM↓</td><td>NISQA↓</td><td>DSR↑</td></tr><tr><td>-</td><td>=</td><td>1.99</td><td>45.00%</td><td>0.31</td><td>3.09</td><td>20.20%</td></tr><tr><td>√</td><td>=</td><td>2.42</td><td>31.80%</td><td>0.29</td><td>2.68</td><td>41.20%</td></tr><tr><td>=</td><td>√</td><td>2.23</td><td>23.00%</td><td>0.27</td><td>2.44</td><td>60.60%</td></tr><tr><td></td><td>w/o Sem-free</td><td>2.28</td><td>26.36%</td><td>0.29</td><td>3.30</td><td>26.80%</td></tr><tr><td>√</td><td>√</td><td>2.22</td><td>23.60%</td><td>0.27</td><td>2.10</td><td>57.80%</td></tr></table>

# 5.3 Ablation Study

Effectiveness of Adversarial Identity Obfuscation We ablate the loss designed for Adversarial Identity Obfuscation, $\mathcal { L } _ { I D }$ and $\mathcal { L } _ { c t x }$ , in Table 2. The results show that $\mathcal { L } _ { I D }$ alone effectively reduces the ASV acceptance rate. This confirms that directly manipulating the representation learning embedding space effectively disrupts recognizable identity features. Removing the opposite-gender guidance from $\mathcal { L } _ { I D }$ worsens ASV, confirming our psychoacoustically-motivated strategy provides effective direction for identity disruption. Separately, the context divergence loss also contributes significantly, lowering both ASV and NISQA by interfering with the attention mechanism’s condition injection.

Table 4: Transferability of the proposed defense method against unseen target models.   

<table><tr><td rowspan="2">Target Models</td><td colspan="5">Defense Effectiveness</td></tr><tr><td>DTW↑</td><td>ASV↓</td><td>SSIM↓</td><td>NISQA↓</td><td>DSR↑</td></tr><tr><td>DiffVC</td><td>2.12</td><td>11.40%</td><td>0.27</td><td>2.36</td><td>71.40%</td></tr><tr><td>DDDM-VC</td><td>1.67</td><td>16.80%</td><td>0.36</td><td>2.79</td><td>54.89%</td></tr><tr><td>DuTa-VC</td><td>2.41</td><td>13.77%</td><td>0.27</td><td>2.14</td><td>73.92%</td></tr></table>

Table 5: Computational overhead analysis. ”Time”: average time to generate one sample. ”Mem.”: peak GPU usage.   

<table><tr><td></td><td>Time (s)</td><td>Mem.(GiB)</td></tr><tr><td>Ours</td><td>148.66</td><td>8.97</td></tr></table>

Effectiveness of Perceptual Fidelity Degradation Table 3 ablates the perceptual quality degradation losses. The results show that $\mathcal { L } _ { s c o r e }$ alone is effective, degrading quality (lower NISQA, higher DTW) by forcing the denoising trajectory away from high-fidelity regions. The semantic corruption loss, $\mathcal { L } _ { s c o r e }$ , demonstrates an even stronger individual impact by directly corrupting internal U-Net features, thereby impairing the model’s reconstruction of coherent, natural-sounding speech details. Critically, the necessity of guiding semantic features towards an incoherent state is evident from the ”w/o Sem-free” variant.

# 5.4 Transferability

To demonstrate applicability, we extended our experiments to include two additional open-source diffusion-based VC models: DuTa-VC Wang et al. (2023) and DDDM-VC Choi, Lee, and Lee (2024). As shown in Table 4, VoiceCloak demonstrate favorable transferability to different models, achieving an average DSR of $6 6 . 7 \%$ . We attribute this transferability to: Targeting Common Vulnerabilities, as our method exploits fundamental mechanisms (e.g. attention, score prediction) that are shared across diffusion VCs.

# 5.5 Robustness

We evaluated the robustness of our method against four common distortions, with results presented in Figure 6. The results show consistent defense effectiveness, indicating resilience to real-world transformations and significantly outperforming the undefended baseline. Further details and analysis are available in Appendix.

# 5.6 Efficiency Discussion

As shown in Table 5, the protection process is a one-time, offline operation, and the required GPU memory is within the

![](images/90e8fb0ee494d73d2744e63f60635c973b365c098b70fda4c90ffc2f3404e243.jpg)  
Figure 6: Resilience of VoiceCloak under more advanced robust scenario.

capacity of consumer-grade hardware, making VoiceCloak a practical defense method.

# 6 Conclusion

This paper introduced VoiceCloak, a comprehensive defense against unauthorized diffusion-based voice cloning (VC). Our framework achieves superior defense effectiveness by exploiting targeted intrinsic vulnerabilities within the diffusion process. Through strategies designed to disrupt attention-based conditional guidance, steer the denoising trajectory, and corrupt internal semantic representations, VoiceCloak effectively undermines the synthesis process. Extensive experiments validate its efficacy, demonstrating success in simultaneously obfuscating speaker identity and degrading audio quality to mitigate the threats of voice mimicry Extensive experiments validate its ability to significantly hinder voice cloning by simultaneously disrupting identity and degrading audio quality, thereby mitigating the threats of high-quality voice mimicry

# References

Chen, S.; Chen, L.; Zhang, J.; Lee, K.; Ling, Z.; and Dai, L. 2024. Adversarial speech for voice privacy protection from personalized speech generation. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 11411–11415. IEEE.   
Chen, S.; Wang, C.; Chen, Z.; Wu, Y.; Liu, S.; Chen, Z.; Li, J.; Kanda, N.; Yoshioka, T.; Xiao, X.; et al. 2022a. Wavlm: Large-scale self-supervised pre-training for full stack speech processing. IEEE Journal of Selected Topics in Signal Processing, 16(6): 1505–1518.   
Chen, S.; Wu, Y.; Wang, C.; Chen, Z.; Chen, Z.; Liu, S.; Wu, J.; Qian, Y.; Wei, F.; Li, J.; et al. 2022b. Unispeechsat: Universal speech representation learning with speaker aware pre-training. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 6152–6156. IEEE. Choi, H.-Y.; Lee, S.-H.; and Lee, S.-W. 2024. Dddm-vc: Decoupled denoising diffusion models with disentangled representation and prior mixup for verified robust voice conversion. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 17862–17870.   
Dong, S.; Chen, B.; Ma, K.; and Zhao, G. 2024. Active Defense Against Voice Conversion through Generative Adversarial Network. IEEE Signal Processing Letters, 31: 706– 710.   
Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33: 6840–6851.   
Huang, C.-y.; Lin, Y. Y.; Lee, H.-y.; and Lee, L.-s. 2021. Defending your voice: Adversarial attack on voice conversion. In 2021 IEEE Spoken Language Technology Workshop (SLT), 552–559. IEEE.   
Jeong, M.; Kim, H.; Cheon, S. J.; Choi, B. J.; and Kim, N. S. 2021. Diff-TTS: A Denoising Diffusion Model for Text-to-Speech. In Proc. Interspeech 2021, 3605–3609.   
Jung, J.-w.; Heo, H.-S.; Tak, H.; Shim, H.-j.; Chung, J. S.; Lee, B.-J.; Yu, H.-J.; and Evans, N. 2022. Aasist: Audio antispoofing using integrated spectro-temporal graph attention networks. In ICASSP 2022-2022 IEEE international conference on acoustics, speech and signal processing (ICASSP), 6367–6371. IEEE.   
Kang, M.; Song, D.; and Li, B. 2023. Diffattack: Evasion attacks against diffusion-based adversarial purification. Advances in Neural Information Processing Systems, 36: 73919–73942.   
Katharopoulos, A.; Vyas, A.; Pappas, N.; and Fleuret, F. 2020. Transformers are rnns: Fast autoregressive transformers with linear attention. In International Conference on Machine Learning, 5156–5165. PMLR.   
Kawar, B.; Zada, S.; Lang, O.; Tov, O.; Chang, H.; Dekel, T.; Mosseri, I.; and Irani, M. 2023. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Cision and Pattern recognition, 6007–6017.   
Kong, Z.; Ping, W.; Huang, J.; Zhao, K.; and Catanzaro, B. 2020. Diffwave: A versatile diffusion model for audio synthesis. arXiv preprint arXiv:2009.09761.   
Kreiman, J.; and Sidtis, D. 2011. Foundations of voice studies: An interdisciplinary approach to voice production and perception. John Wiley & Sons.   
Kubichek, R. 1993. Mel-cepstral distance measure for objective speech quality assessment. In Proceedings of IEEE pacific rim conference on communications computers and signal processing, volume 1, 125–128. IEEE.   
Kullback, S.; and Leibler, R. A. 1951. On information and sufficiency. The annals of mathematical statistics, 22(1): 79–86.   
Lavner, Y.; Gath, I.; and Rosenhouse, J. 2000. The effects of acoustic modifications on the identification of familiar voices speaking isolated vowels. Speech Communication, 30(1): 9–26. Li, J.; Ye, D.; Tang, L.; Chen, C.; and Hu, S. 2023. Voice Guard: Protecting Voice Privacy with Strong and Imperceptible Adversarial Perturbation in the Time Domain. In IJ-CAI, 4812–4820.   
Liu, H.; Chen, Z.; Yuan, Y.; Mei, X.; Liu, X.; Mandic, D.; Wang, W.; and Plumbley, M. D. 2023. AudioLDM: Text-to-Audio Generation with Latent Diffusion Models. In International Conference on Machine Learning, 21450–21474. PMLR.   
Oh, H.-S.; Lee, S.-H.; and Lee, S.-W. 2024. Diffprosody: Diffusion-based latent prosody generation for expressive speech synthesis with prosody conditional adversarial training. IEEE/ACM Transactions on Audio, Speech, and Language Processing.   
Popov, V.; Vovk, I.; Gogoryan, V.; Sadekova, T.; and Kudinov, M. 2021. Grad-tts: A diffusion probabilistic model for text-to-speech. In International Conference on Machine Learning, 8599–8608. PMLR.   
Popov, V.; Vovk, I.; Gogoryan, V.; Sadekova, T.; Kudinov, M. S.; and Wei, J. 2022. Diffusion-Based Voice Conversion with Fast Maximum Likelihood Sampling Scheme. In International Conference on Learning Representations.   
Rix, A. W.; Beerends, J. G.; Hollier, M. P.; and Hekstra, A. P. 2001. Perceptual evaluation of speech quality (PESQ)- a new method for speech quality assessment of telephone networks and codecs. In 2001 IEEE InternationalConference on Acoustics, Speech, and Signal Processing. Proceedings (Cat. No. 01CH37221), volume 2, 749–752. IEEE. Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Ommer, B. 2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern recognition, 10684– 10695.   
Ronneberger, O.; Fischer, P.; and Brox, T. 2015. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, 234–241. Springer.   
Ruan, L.; Ma, Y.; Yang, H.; He, H.; Liu, B.; Fu, J.; Yuan, N. J.; Jin, Q.; and Guo, B. 2023. Mm-diffusion: Learning multi-modal diffusion models for joint audio and video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10219–10228. Ruiz, N.; Li, Y.; Jampani, V.; Pritch, Y.; Rubinstein, M.; and Aberman, K. 2023. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 22500–22510.   
Sakoe, H.; and Chiba, S. 1978. Dynamic programming algorithm optimization for spoken word recognition. IEEE transactions on acoustics, speech, and signal processing, 26(1): 43–49.   
Shen, K.; Ju, Z.; Tan, X.; Liu, E.; Leng, Y.; He, L.; Qin, T.; Bian, J.; et al. 2024. NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers. In The Twelfth International Conference on Learning Representations.   
Song, J.; Meng, C.; and Ermon, S. 2020. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502.   
Song, Y.; Sohl-Dickstein, J.; Kingma, D. P.; Kumar, A.; Ermon, S.; and Poole, B. 2020. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456.   
Tumanyan, N.; Geyer, M.; Bagon, S.; and Dekel, T. 2023. Plug-and-play diffusion features for text-driven image-toimage translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1921– 1930.   
Wang, F.; Tan, Z.; Wei, T.; Wu, Y.; and Huang, Q. 2024. Simac: A simple anti-customization method for protecting face privacy against text-to-image synthesis of diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 12047–12056. Wang, H.; Thebaud, T.; Villalba, J.; Sydnor, M.; Lammers, B.; Dehak, N.; and Moro-Velazquez, L. 2023. DuTa-VC: A Duration-aware Typical-to-atypical Voice Conversion Approach with Diffusion Probabilistic Model. In Proc. Interspeech 2023, 1548–1552.   
Wu, J.; Lu, W.; Luo, X.; Yang, R.; Wang, Q.; and Cao, X. 2024. Coarse-to-fine proposal refinement framework for audio temporal forgery detection and localization. In Proceedings of the 32nd ACM International Conference on Multimedia, 7395–7403.   
Yamagishi, J.; Veaux, C.; and MacDonald, K. 2019. CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version 0.92).   
Yu, H.; Chen, J.; Ding, X.; Zhang, Y.; Tang, T.; and Ma, H. 2024. Step vulnerability guided mean fluctuation adversarial attack against conditional diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 6791–6799.   
Yu, Z.; Zhai, S.; and Zhang, N. 2023. Antifake: Using adversarial audio to prevent unauthorized speech synthesis. In Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security, 460–474.   
Zen, H.; Dang, V.; Clark, R.; Zhang, Y.; Weiss, R. J.; Jia, Y.; Chen, Z.; and Wu, Y. 2019. Libritts: A corpus derived from librispeech for text-to-speech. arXiv preprint arXiv:1904.02882.   
Zhou, Y.; and Lim, S.-N. 2021. Joint audio-visual deepfake detection. In Proceedings of the IEEE/CVF International Conference on Computer Cision, 14800–14809.

# Technical Appendix

# A Additional Explanation on Our Strategy

Algorithm. The full procedure of our multi-dimensional defense is detailed in Algorithm 1

Speaker Embedding: This strategy targets a pre-trained d-vector as speaker encoder.

• VoicePrivacy Chen et al. (2024): For this baseline, we strictly follow the original implementation from the original paper. Notably, this setup differs from the primary configuration used for our main experiments. Specifically, adversarial perturbations are generated using the Fast Gradient Sign Method (FGSM) with a prescribed budget of $\epsilon \ : = \ : 0 . 0 2$ . The targeted speaker encoder, as specified in their work, is the pre-trained ECAPA-TDNN model provided with YourTTS. And the optimization objective is to maximize the corresponding cosine similarity between extracted embeddings.

• VoiceGuard Li et al. (2023): VoiceGuard optimizes timedomain perturbations and uniquely constrains their magnitude using a psychoacoustic model rather than a fixed $l _ { p }$ -norm budget. Given this distinct constraint mechanism, a direct budget comparison is not applicable. Therefore, for a fair evaluation consistent with its design, we employed the default hyperparameter settings provided in the official codebase.

$$
s c o r e \gets s _ { \theta } \big ( \mathbf { x } _ { r e f } , \mathbf { x } _ { a d v } , T _ { a d v } \big )
$$

$$
\begin{array} { r l } & { \mathcal { L } _ { I D } ^ { ' } \gets 1 - \mathsf { S I M } ( \mathbf { e } _ { a d v } , \mathbf { e } _ { r e f } ) + \mathsf { S I M } ( \mathbf { e } _ { a d v } , \mathbf { C } _ { o p p } ) } \\ & { P _ { r e f } \gets \mathsf { S o f t m a x } ( \mathcal { H } _ { c t x } ( \mathbf { x } _ { r e f } ) ) } \\ & { P _ { a d v } \gets \mathrm { S o f t m a x } ( \mathcal { H } _ { c t x } ( \mathbf { x } _ { a d v } ) ) } \\ & { \mathcal { L } _ { c t x } \gets D _ { K L } ( P _ { r e f } \parallel P _ { a d v } ) } \\ & { \mathcal { L } _ { s c o r e } \gets | | s c o r e | | _ { 2 } } \\ & { \mathcal { L } _ { s e m } \gets ( 1 - \cos ( \mathbf { f } _ { a d v } , \mathbf { f } _ { r e f } ) ) + \cos ( \mathbf { f } _ { a d v } , \mathbf { f } _ { n o i s e } ) } \end{array}
$$

$$
\begin{array} { r l } & { \quad \bar { \mathcal { L } } _ { t o t a l }  \lambda _ { I D } \mathcal { L } _ { I D } + \bar { \lambda _ { c t x } } \mathcal { L } _ { c t x } + \lambda _ { s c o r e } \mathcal { L } _ { s c o r e } + } \\ & { s e m \mathcal { L } _ { s e m } } \\ & { \quad { \bf x } _ { a d v }  { \bf x } _ { a d v } + \alpha \cdot \mathrm { s i g n } ( \nabla _ { { \bf x } _ { \bf a d v } } \mathcal { L } _ { { \bf t o t a l } } ) } \\ & { \quad { \bf x } _ { a d v }  { \bf x } _ { r e f } + \mathrm { c l a m p } ( { \bf x } _ { a d v } - { \bf x } _ { r e f } , - \epsilon , \epsilon ) } \end{array}
$$

# B.2 Evaluation Metrics

To provide a comprehensive and multi-faceted evaluation of VoiceCloak, we employ a suite of objective metrics assessing three key aspects: identity obfuscation, perceptual quality degradation, and perturbation imperceptibility. Let $y$ denote the audio synthesized using the original reference $x _ { r e f }$ , and $y _ { a d v }$ denote the audio synthesized using the protected reference $x _ { a d v }$ .

# B Experimental Details

# B.1 Baseline Methods

To ensure a fair comparison, all baseline methods were configured in a white-box setting. For methods with official implementations (e.g., Attack-VC, VoiceGuard), we adapted their public code for our experimental pipeline. Perturbation budgets and key parameters for each baseline were carefully set, adhering to their original papers where possible.

• Random Noise: A naive baseline where Gaussian noise $( x \sim \mathcal { N } ( 0 , I ) )$ is added to the reference audio. The noise is clipped to the same $l _ { \infty }$ -norm budget as our method for a direct comparison of structured vs. unstructured noise. • Attack-VC Huang et al. (2021): Following their paper, we combined two of their primary strategies: End-to-End (e2e): This defense directly maximizes the Melspectrogram $l _ { 2 }$ distance between the VC outputs generated with the original and perturbed references which backpropagated through the entire Diffusion model.

Automatic Speaker Verification (ASV): This metric quantifies the core goal of identity obfuscation. We utilize a pre-trained, state-of-the-art ECAPA-TDNN Desplanques, Thienpondt, and Demuynck (2020) model as our speaker verification system. For each target speaker, we first compute an average enrollment embedding from set of their clean, original utterances. Then, for each defended output $y _ { a d v }$ , we extract its embedding and compute a cosine similarity score against the corresponding target speaker’s enrollment embedding. The ASV Acceptance Rate is samples whose similarity score exceeds a pre-defined threshold, which can be formulated as: $S i m ( \bar { y _ { a d v } } , a v g ( X _ { r e f } ) ) \geq$ $\tau _ { A S V }$ .

NISQA: NISQA Mittag and Moller (2021) is a neural ¨ network-based model for objective audio quality assessment. It is trained to predict the subjective Mean Opinion Score (MOS) that a human listener would assign, evaluating the overall quality and naturalness of speech on a scale of 1 to 5. By leveraging a deep model trained on human perception data, NISQA provides a robust and objective proxy for subjective listening tests.

Dynamic Time Warping: DTW Sakoe and Chiba (1978) measures the distance between the temporal structures of two audio signals. Based on our implementation, we first extract the Mel spectrograms of the undefended $y$ and defended $y _ { a d v }$ audio outputs. We then compute the cityblock distance between these two spectrograms using the fastdtw algorithm to find the optimal alignment path.

Structural Similarity (SSIM): Originally an image quality metric, we adapt SSIM Wang et al. (2004) to measure the perceptual similarity between the spectrograms of $y$ and $y _ { a d v }$ . We first compute the Short-Time Fourier Transform (STFT) of both audio signals and convert the resulting magnitudes to a log-dB scale. These log-magnitude spectrograms are then normalized to a range of [0, 1]. The SSIM score is calculated between these normalized spectrograms. The SSIM index between two spectrograms is calculated as:

$$
\mathrm { S S I M } ( u , v ) = \frac { ( 2 \mu _ { u } \mu _ { v } + C _ { 1 } ) ( 2 \sigma _ { u v } + C _ { 2 } ) } { ( \mu _ { u } ^ { 2 } + \mu _ { v } ^ { 2 } + C _ { 1 } ) ( \sigma _ { u } ^ { 2 } + \sigma _ { v } ^ { 2 } + C _ { 2 } ) } ,
$$

where $\mu$ and $\sigma$ represent the mean and standard deviation, respectively, $\sigma _ { u v }$ is the covariance, and $C _ { 1 } , C _ { 2 }$ are stabilizing constants. A lower SSIM score (closer to 0) indicates less structural similarity and thus a greater defensive effect.

Defense Success Rate (DSR): DSR is our primary comprehensive metric, designed to holistically evaluate the achievement of our two defense objectives. A defense instance is deemed successful only if the resulting audio $y _ { a d v }$ simultaneously meets two conditions: (1) it fails the speaker verification test (i.e., ASV score $\le \tau _ { A S V }$ ), and (2) it exhibits low perceptual quality. Eventually, a successful defense is formally defined as:

$$
\mathrm { S u c c e s s } = ( \mathrm { A S V } ( y _ { a d v } ) < \tau _ { A S V } ) \wedge ( \mathrm { N I S Q A } ( y _ { a d v } ) < \tau _ { q } ) ,
$$

where we set the quality threshold $\tau _ { q } ~ = ~ 0 . 3 0$ and identity threshold $\tau _ { A S V } ~ = ~ 0 . 2 5$ . The thresholds were chosen to reflect reasonable boundaries for identity verification and quality assessment. And these choices align with prior work or related paper, providing a consistent standard and quantifiable success defense.

The following metrics evaluate the quality of the adversarial reference audio $x _ { a d v }$ itself, by comparing it to the original $x _ { r e f }$ .

Perceptual Evaluation of Speech Quality (PESQ): PESQ Rix et al. (2001) is a classic objective metric that predicts the subjective quality of speech. We use it to measure the degradation introduced by the perturbation. Scores range from -0.5 to 4.5, with higher scores indicating better quality (i.e., a more imperceptible perturbation).

Mel-Cepstral Distortion (MCD): This calculates the Euclidean distance between the Mel-frequency cepstral coefficients (MFCCs) of $x _ { r e f }$ and $x _ { a d v }$ . It is a standard metric for measuring spectral distortion in speech. A lower MCD value indicates less spectral distortion and thus a more imperceptible perturbation.

Signal-to-Noise Ratio: SNR measures the ratio of the power of the original signal $x _ { r e f }$ to the power of the perturbation noise $\delta$ . It is calculated in decibels (dB) as:

$$
\mathrm { S N R _ { d B } } = 1 0 \log 1 0 \left( \frac { \sum _ { n } x _ { r e f } ( n ) ^ { 2 } } { \sum _ { n } \delta ( n ) ^ { 2 } } \right)
$$

where the sums are over all samples in the signal. A higher SNR value indicates a weaker perturbation relative to the signal.

Table 6: Hyperparameters used for the PGD.   

<table><tr><td>Norm</td><td>E</td><td> step size α</td><td> optimization iterations</td></tr><tr><td>l8</td><td>0.002</td><td>4×10-5</td><td>50</td></tr></table>

# B.3 Implementation Details

Hyperparameter Setup All audio data was uniformly resampled to $2 2 0 5 0 ~ \mathrm { H z }$ and normalized to the range [-1, 1]. To generate the set of protected audio samples $x _ { a d v }$ , we selected a diverse and gender-balanced subset of reference utterances. Specifically, from LibriTTS, we used 479 utterances, and from VCTK, we used 500 utterances. In both cases, the selection ensures an approximate $50 \%$ male and $50 \%$ female speaker distribution. For the voice cloning synthesis process, all voice cloning evaluations were conducted as gender-matched conversions (i.e., male source to male target, female source to female target), as this setup typically yields the highest baseline voice conversion quality, to ensure a consistent and challenging evaluation scenario.

Table 7: Hyperparameters used for diffusion process.   

<table><tr><td>mel bins</td><td>hidden dims</td><td>denoise timesteps</td><td> Adv. timesteps</td></tr><tr><td>80</td><td>256</td><td>100</td><td>6</td></tr></table>

In this paper, our adversarial perturbations are generated using the Projected Gradient Descent (PGD) algorithm. The optimization process is configured with the hyperparameters detailed in Table 6. For the diffusion process itself, we set the configuration with the hyperparameters, as shown in Table 7.

Human Evaluation We conducted a subjective human evaluation study to assess the real-world perceptual impact of VoiceCloak compared to baseline methods. We recruited 50 participants for the study. A total of 100 distinct audio clips were used for evaluation, comprising samples synthesized using references protected by our method and the primary baseline methods. All participants were asked to use headphones in a quiet environment to ensure a consistent and high-quality listening experience. Participants were then asked to make a comparative judgment in response to the following two questions: (1) ”Which of the two synthesized voices sounds less similar to the original speaker’s voice?” This question directly measures the perceived effectiveness of identity obfuscation. (2) ”Which of the two synthesized voices sounds less natural or more distorted?” This question evaluates the perceived degree of perceptual quality degradation. For each question, participants could choose one of the two synthesized audio clips or select ”Neutral” if they perceived no significant difference. To mitigate ordering bias, the positions of the two synthesized samples were randomly shuffled for each question and each participant.

<table><tr><td rowspan="2">Lossy Type</td><td rowspan="2">Level(↑)</td><td colspan="4">Defense Effectiveness</td></tr><tr><td>DTW(↓)</td><td>ASV(↓)</td><td>NISQA(↓)</td><td>DSR(↑)</td></tr><tr><td rowspan="5">AAC Compression</td><td>128</td><td>2.33</td><td>16.70%</td><td>2.46</td><td>65.01%</td></tr><tr><td>96</td><td>2.33</td><td>14.08%</td><td>2.40</td><td>70.65%</td></tr><tr><td>64</td><td>2.31</td><td>15.49%</td><td>2.52</td><td>62.40%</td></tr><tr><td>48</td><td>2.30</td><td>19.92%</td><td>2.57</td><td>60.59%</td></tr><tr><td>32</td><td>2.32</td><td>14.08%</td><td>2.71</td><td>55.76%</td></tr><tr><td rowspan="5">MP3 Compression</td><td>128</td><td>2.33</td><td>17.71%</td><td>2.42</td><td>66.42%</td></tr><tr><td>96</td><td>2.33</td><td>13.48%</td><td>2.41</td><td>69.84%</td></tr><tr><td>64</td><td>2.33</td><td>15.29%</td><td>2.45</td><td>67.03%</td></tr><tr><td>48</td><td>2.31</td><td>17.91%</td><td>2.57</td><td>59.38%</td></tr><tr><td>32</td><td>2.34</td><td>15.29%</td><td>3.19</td><td>29.40%</td></tr><tr><td rowspan="4">Gaussian Noise</td><td>30 25</td><td>2.34 2.35</td><td>19.72%</td><td>2.68</td><td>52.74%</td></tr><tr><td></td><td></td><td>18.51%</td><td>2.76</td><td>50.93%</td></tr><tr><td>20</td><td>2.37</td><td>16.50%</td><td>2.79</td><td>52.94%</td></tr><tr><td>15 10</td><td>2.39 2.47</td><td>12.88% 10.66%</td><td>2.79 2.79</td><td>53.14%</td></tr><tr><td rowspan="5">Lowpass</td><td>7000</td><td>2.32</td><td>13.88%</td><td>2.22</td><td>56.76% 77.89%</td></tr><tr><td>6000</td><td>2.40</td><td>17.30%</td><td>2.66</td><td>62.60%</td></tr><tr><td>5000</td><td>2.39</td><td>21.13%</td><td>2.40</td><td>67.43%</td></tr><tr><td>4000</td><td>2.39</td><td>22.33%</td><td>2.40</td><td>63.81%</td></tr><tr><td>3000</td><td>2.44</td><td>17.91%</td><td>2.56</td><td>65.62%</td></tr></table>

Table 8: Overall robustness of our defense method against various lossy post-processing operations. Lower values for “Level” indicate more severe operations.

# C Additional Experiments

# C.1 Robustness Evaluation

We evaluated the robustness of our adversarial perturbations against common audio distortions encountered during digital transmission and storage. We subjected the protected reference audio $x _ { a d v }$ to four types of transformations with varying levels of severity:

• AAC and MP3 Compression: These are widely used   
lossy audio codecs that reduce file size by discarding less   
perceptible acoustic information. We tested a range of bi  
trates from 128kbps (high quality) down to 32kbps (low   
quality).   
• Gaussian Noise: This involves adding white Gaussian noise to the audio signal, a common way to simulate   
channel noise or environmental interference. We evaluated across Signal-to-Noise Ratios (SNR) from 30dB   
(low noise) to 10dB (high noise).   
• Low-pass Filtering: This process removes high  
frequency components from the signal, which can oc  
cur during resampling or transmission over band-limited   
channels. We tested cutoff frequencies from 7kHz down to 3kHz.

The defense performance under these conditions is detailed in Table 8. Even under moderate degradation, the core defensive properties of the perturbation are well-preserved.

A notable phenomenon occurs under severe distortion (e.g., 32kbps MP3 compression), where defense metrics paradoxically appear to improve. We attribute this not to an enhanced perturbation, but to the degradation of the reference audio’s fundamental signal integrity. At such high distortion levels, essential acoustic features of the speaker are damaged alongside the perturbation. Consequently, the voice cloning model itself is inherently compromised by the poor input quality, struggling to synthesize a coherent, identity-consistent output. In these extreme cases, the high DSR reflects not only our method’s robustness but also the diffusion model’s own limitation when presented with severely degraded inputs.

Table 9: Comparison with prior defense methods under the most severe lossy post-processing conditions.   

<table><tr><td rowspan="2">Lossy Type</td><td rowspan="2">Methods</td><td colspan="3">Defense Effectiveness</td></tr><tr><td>ASV(↓)</td><td>NISQA(↓)</td><td>DSR(↑)</td></tr><tr><td rowspan="3">AAC-32</td><td>Attack-VC</td><td>23.54%</td><td>3.30</td><td>25.35%</td></tr><tr><td>VoicePrivacy</td><td>22.74%</td><td>3.28</td><td>24.35%</td></tr><tr><td>Ours</td><td>14.08%</td><td>2.71</td><td>55.76%</td></tr><tr><td rowspan="3">MP3-32</td><td>Attack-VC VoicePrivacy</td><td>21.73%</td><td>3.60</td><td>12.07%</td></tr><tr><td></td><td>19.52%</td><td>3.66</td><td>11.47%</td></tr><tr><td>Ours</td><td>15.29%</td><td>3.19</td><td>29.40%</td></tr><tr><td rowspan="3">Gaussian-10</td><td>Attack-VC VoicePrivacy</td><td>16.50%</td><td>2.79</td><td>56.14%</td></tr><tr><td>Ours</td><td>16.90%</td><td>2.77</td><td>57.14%</td></tr><tr><td></td><td>10.66%</td><td>2.79</td><td>56.76%</td></tr><tr><td rowspan="3">Lowpass-3k</td><td>Attack-VC</td><td>25.15%</td><td>2.87</td><td>43.69%</td></tr><tr><td>VoicePrivacy</td><td>23.94%</td><td>2.89</td><td>47.89%</td></tr><tr><td>Ours</td><td>17.91%</td><td>2.56</td><td>65.62%</td></tr></table>

Table 10: Comparison of computational overhead for our method and baseline.   

<table><tr><td>Methods</td><td>Time (seconds/audio)</td><td>Peak Memory (GB)</td></tr><tr><td>Attack-VC</td><td>181.35</td><td>9.80</td></tr><tr><td>VoicePrivacy</td><td>16.12</td><td>0.61</td></tr><tr><td>VoiceGuard</td><td>248.91</td><td>14.36</td></tr><tr><td>Ours</td><td>148.66</td><td>9.19</td></tr></table>

Furthermore, Table 9 compares the robustness of Voice-Cloak against baseline methods under the most severe distortion conditions. The results show that VoiceCloak consistently demonstrates superior defense effectiveness. Notably, VoiceCloak also achieves the lowest ASV acceptance rate in all scenarios, indicating that the identity obfuscation component of our perturbation is particularly resilient to these transformations. This superior robustness is likely attributed to our multi-dimensional defense strategy. While simple post-processing might partially neutralize attacks targeting a single vulnerability (e.g., only the embedding space), Voice-Cloak’s approach of simultaneously corrupting conditional guidance, the denoising trajectory, and internal semantic features creates a more resilient and multifaceted defense that is harder to fully compromise.

# C.2 Computational Costs

The practical utility of a proactive defense method is closely tied to its computational cost. For a tool like VoiceCloak to be accessible to individual users, it must be both timeefficient enough for practical use and memory-efficient enough to run on common consumer-grade hardware. In this section, we analyze the computational overhead of our method and compare it against the baselines.

Analysis of the Computational Cost in VoiceCloak Time Cost. In each of the 50 iterations, the primary operations are: (1) a forward pass through the diffusion model to compute the intermediate features and the score prediction $s _ { \theta }$ , which is necessary for all our loss components; and (2) a backward pass to compute the gradient of the total loss $\mathcal { L } _ { t o t a l }$ with respect to the input audio $x _ { a d v }$ . We set the adversarial optimization iterations to be 50 and the gradient repeats to be 5 for an average gradient. Therefore, we run $5 0 \times 5 = 2 5 0$ steps for one input optimization. As we select the early stage of diffusion steps, it would be time-efficient for backward propagation.

Memory Cost. The peak GPU memory usage during this process consists of three main components: (1) the model weights of the DiffVC model and any auxiliary models (like WavLM), which must be loaded into VRAM; (2) the computation graph and intermediate activations stored during the forward pass, which are necessary for gradient computation in the backward pass; and (3) a smaller amount of memory for the input audio tensors and the calculated gradients themselves. The dominant factor for memory is the size of the computation graph, which is directly related to the depth and complexity of the DiffVC U-Net.

Evaluation on Costs Table 10 presents a comparison of the average time required to process a single audio sample (approximately 7 seconds in length) for our method and the baselines on an NVIDIA RTX 3090 GPU. VoiceCloak requires approximately 148.7 seconds per sample. This is significantly faster than VoiceGuard (248.9s) and Attack-VC (181.4s). While methods like VoicePrivacy (16.1s) are faster, they often target simpler objectives and may not provide the same level of multi-faceted defense. Crucially, as this is a one-time, offline pre-processing step, we argue that this time cost is a reasonable and practical investment for users seeking to permanently protect their audio data before dissemination.

The peak GPU memory usage is another critical factor for accessibility. The memory cost of VoiceCloak is primarily determined by the storage of the DiffVC model weights and the computation graph required for backpropagation. A key design choice that significantly enhances our memory efficiency is the targeted computation over early denoising timesteps. Unlike a hypothetical, fully end-to-end attack that would need to backpropagate through the entire multi-step denoising trajectory—a process requiring prohibitively large memory to store dozens of U-Net computation graphs (at least $8 \times$ Nvidia A100 with 40GB memory). By concentrating gradient computations on only a few initial steps $T _ { a d v } = 6 )$ ), VoiceCloak only needs to store a much shallower computation graph, dramatically reducing the memory footprint. As shown in Table 10, the peak memory usage of VoiceCloak is approximately 9.19 GB. This is lower than both Attack-VC (9.80 GB) and Voice-Guard (14.36 GB), which require larger memory footprints. A memory requirement of $9 . 2 \ : \mathrm { G B }$ ensures that VoiceCloak is well within the capacity of most modern consumer-grade GPUs (e.g., those with 10GB, 12GB, or more VRAM), making it a highly accessible tool for individual users and nonspecialists. This practical memory efficiency is a key advantage for real-world deployment.

# C.3 Additional Ablation Studies

Perturbation budgets We investigated the influence of the perturbation budget $\epsilon$ on both defense effectiveness and adversarial imperceptibility, with results summarized in Table 11. The findings reveal a clear trade-off: increasing e directly enhances defense effectiveness but conversely degrades audio imperceptibility. The excessive distortion compromises the usability of $x _ { a d v }$ for legitimate purposes. We determined $\epsilon = 0 . 0 0 2 0$ to be a suitable default budget, providing strong protection while preserving acceptable adversarial quality.

<table><tr><td rowspan="2">Budgets (e)</td><td colspan="5">Defense Effectiveness</td><td colspan="3">Imperceptibility</td></tr><tr><td>DTW(↑)</td><td>ASV(↓)</td><td>SSIM(↓)</td><td>NISQA(↓)</td><td>DSR(↑)</td><td>PESQ(↑)</td><td>MCD(↓)</td><td>SNR(↑)</td></tr><tr><td>Undefended</td><td>-</td><td>76.49%</td><td>1</td><td>3.96</td><td>=</td><td>1</td><td>1</td><td>1</td></tr><tr><td>0.0005</td><td>2.01</td><td>22.00%</td><td>0.29</td><td>2.86</td><td>41.20%</td><td>4.25</td><td>0.48</td><td>45.03</td></tr><tr><td>0.0010</td><td>2.05</td><td>16.60%</td><td>0.28</td><td>2.65</td><td>57.00%</td><td>3.81</td><td>0.79</td><td>39.26</td></tr><tr><td>0.0020</td><td>2.12</td><td>11.40%</td><td>0.27</td><td>2.36</td><td>71.20%</td><td>3.21</td><td>1.29</td><td>33.53</td></tr><tr><td>0.0050</td><td>2.23</td><td>6.20%</td><td>0.25</td><td>1.95</td><td>84.00%</td><td>2.34</td><td>2.34</td><td>25.94</td></tr><tr><td>0.0100</td><td>2.32</td><td>1.20%</td><td>0.24</td><td>1.75</td><td>90.80%</td><td>1.74</td><td>3.40</td><td>20.18</td></tr></table>

Table 11: Performance of VoiceCloak under different perturbation budgets. ”Undefended” denotes voice conversion without VoiceCloak.

Robust to Inference Steps In a practical defense scenario, the defender cannot know the specific number of inference steps $( T )$ an attacker will use for the voice cloning synthesis. To evaluate the robustness of VoiceCloak against this uncertainty, we conducted an ablation study where the attacker’s total number of inference timesteps was varied. For this experiment, the gradient-based loss components for perturbation were computed only within the early diffusion timesteps $T _ { a d v } = 6 )$ ). We then evaluated the effectiveness of this single, fixed perturbation when the diffusion model used it to synthesize audio over different inference path lengths, from a very short $T = 6$ to $T = 2 0 0$ . The results are presented in Table 12.

Table 12: Ablation study on the effect of different diffusion inference steps $T$ . ”Avg. RT” indicates the average runtime per generated sample.   

<table><tr><td rowspan="2">Infer. Steps T|</td><td colspan="4">Defense Effectiveness</td><td rowspan="2">Avg.RT (s)</td></tr><tr><td>DTW(↑)</td><td>ASV(↓)</td><td>NISQA(↓)</td><td>DSR(↑)</td></tr><tr><td>6</td><td>1.72</td><td>11.80%</td><td>1.83</td><td>84.20%</td><td>0.32</td></tr><tr><td>18</td><td>1.94</td><td>10.40%</td><td>2.08</td><td>81.80%</td><td>0.71</td></tr><tr><td>30</td><td>2.03</td><td>11.40%</td><td>2.13</td><td>78.80%</td><td>1.09</td></tr><tr><td>100</td><td>2.13</td><td>11.00%</td><td>2.37</td><td>71.80%</td><td>3.28</td></tr><tr><td>200</td><td>2.15</td><td>13.40%</td><td>2.37</td><td>69.20%</td><td>6.57</td></tr></table>

The results demonstrate that VoiceCloak exhibits remarkable stability, maintaining high defense effectiveness across a wide range of attacker inference steps. While minor fluctuations exist, key metrics like DSR and ASV remain consistently strong, indicating that the core defense is not compromised by the length of the synthesis process. This stability stems directly from our design choice to concentrate adversarial optimization on the initial denoising timesteps $( T _ { a d v } = 6 )$ ) that early steps are critical for reconstructing the fundamental low-frequency and coarse structural components of the audio signal. By introducing potent disruptions at this foundational stage, the errors are irrevocably embedded into the generation process. Consequently, any subsequent refinement steps operate on an already corrupted foundation. This is evidenced by the trend in the DTW score, which increases with $T$ . This suggests that as the model performs more refinement steps, the initial error is not corrected but rather cumulatively amplified.

Ablation on PGD Iterations We conducted an ablation study to analyze the impact of the number of PGD optimiza-

tion iterations on both defense effectiveness and computational efficiency. Table 13 presents the results as we vary the number of iterations from 5 to 100.   
Table 13: Ablation study on the number of adversarial attack iterations. Avg Time is measured in seconds per sample.   

<table><tr><td>PGD Iters</td><td>Avg Time/sample(s)</td><td>ASV(↓)</td><td>NISQA(↓)</td><td>DSR(↑)</td></tr><tr><td>5</td><td>15.02</td><td>0.178</td><td>2.63</td><td>0.564</td></tr><tr><td>10</td><td>29.81</td><td>0.142</td><td>2.46</td><td>0.636</td></tr><tr><td>25</td><td>74.94</td><td>0.110</td><td>2.32</td><td>0.734</td></tr><tr><td>50</td><td>148.66</td><td>0.128</td><td>2.33</td><td>0.716</td></tr><tr><td>100</td><td>295.80</td><td>0.100</td><td>2.26</td><td>0.720</td></tr></table>

The results reveal a clear relationship between the number of iterations and the performance. As expected, the average time cost increases linearly with the number of PGD iterations. In terms of defense effectiveness, we observe that performance, particularly the Defense Success Rate (DSR), generally improves as the number of iterations increases from 5 to 25. However, a notable point of diminishing returns is observed beyond this. The DSR peaks at $7 3 . 4 \%$ with 25 iterations and then slightly plateaus or fluctuates around a similar level for 50 and 100 iterations $7 1 . 6 \%$ and $7 2 . 0 \%$ respectively). This indicates that while the time cost continues to grow linearly, the defense performance reaches a bottleneck after approximately 25-50 iterations. This analysis highlights the flexibility of VoiceCloak and informs the optimal choice of parameters based on different application needs:

• For security-critical scenarios where maximizing defense is paramount, setting the iteration count to 50 provides a near-optimal level of protection, as further increases yield minimal performance gains at a significant additional time cost.   
• For efficiency-critical scenarios where faster processing is required, setting the iteration count to 25 offers an excellent trade-off, providing strong defense effectiveness at roughly 1min.

# D Additional Related Work

# D.1 Adversarial Examples for Classifiers

The paradigm of adversarial examples was first established in the domain of image classification Goodfellow et al. (2014); Madry et al. (2017a); Carlini and Wagner (2017).

The core objective is to generate an input, $x ^ { \prime }$ , that is perceptually similar to a benign input $x$ but causes a trained classifier, $f ( \cdot )$ , to produce an incorrect prediction. Formally, an adversarial example $x ^ { \prime }$ is defined by two primary properties: 1. Imperceptibility: The perturbation between $x$ and $x ^ { \prime }$ is minimal, such that $\mathcal { D } ( x , x ^ { \prime } ) \leq \epsilon$ for a given distance metric $\mathcal { D }$ and a small buget $\epsilon$ . Consistent with prior work, the $l _ { \infty }$ -norm is a prevalent metric for this constraint. 2. Effectiveness: The model’s prediction for the adversarial example is incorrect, i.e., $f ( x ^ { \prime } ) \neq y$ , where $y$ is the ground-truth label for $x$ .

This is typically framed as a constrained optimization problem, where the goal is to find a perturbation $\delta$ that maximizes a loss function $\mathcal { L }$ encouraging misclassification, subject to the imperceptibility constraint:

$$
\delta _ { a d v } = \arg \operatorname* { m a x } _ { | | \delta | | _ { p } \leq \epsilon } \mathcal { L } ( f ( x + \delta ) , y ) .
$$

This problem is commonly solved using iterative methods, with Projected Gradient Descent (PGD) Madry et al. (2017b) being a standard and robust algorithm. PGD iteratively updates the adversarial example by taking a small step in the direction of the loss gradient and then projecting the result back onto the $\epsilon$ -ball to satisfy the constraint:

$$
\begin{array} { r } { x _ { r + 1 } = \Pi _ { \mathbf { x } , \epsilon } ( \mathbf { x } _ { r } + \alpha \cdot \mathrm { s i g n } ( \nabla _ { \mathbf { x } } \mathcal { L } ( ( f ( \mathbf { x } _ { r } ) , y ) ) ) , } \end{array}
$$

where $r$ denotes the iteration, $\alpha$ is the projection operator ensuring that the updated example ${ \bf x } _ { r + 1 }$ remains within the $\epsilon$ -ball around the original input $\mathbf { x }$ .

# D.2 Adversarial Attack for Deepfake

The principles of adversarial examples have been extended from classifiers to generative models, such as those used for Deepfake synthesis. In this domain, the attack objective shifts from causing misclassification to degrading the quality of the generated output or disrupting the synthesis process entirely. One primary strategy involves attacking ancillary components within the generation pipeline, such as facial landmark detectors or feature extractors, which are often essential pre-processing steps for deepfake models. By corrupting the inputs to these upstream modules, the final output of the core generative model is consequently compromised. This approach is analogous to early proactive audio defenses that targeted speaker encoders Huang et al. (2021). Another more direct strategy targets the core generative model itself. While initial efforts focused on exploiting architectural properties of GANs to degrade image quality, these methods are often ineffective against the distinct mechanisms of Diffusion Models (DMs). For DMs, an effective adversarial example is one that the model considers ”out-of-distribution”, thereby hindering its superior reconstruction capabilities. Our work builds upon this more direct approach. However, it is specifically tailored to the unique vulnerabilities of DMs (e.g., attention, score function, U-Net features), addressing the limitations of defenses designed for prior generative paradigms.

# References

Carlini, N.; and Wagner, D. 2017. Towards evaluating the robustness of neural networks. In 2017 ieee symposium on security and privacy (sp), 39–57. Ieee.   
Chen, S.; Chen, L.; Zhang, J.; Lee, K.; Ling, Z.; and Dai, L. 2024. Adversarial speech for voice privacy protection from personalized speech generation. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 11411–11415. IEEE.   
Desplanques, B.; Thienpondt, J.; and Demuynck, K. 2020. Ecapa-tdnn: Emphasized channel attention, propagation and aggregation in tdnn based speaker verification. arXiv preprint arXiv:2005.07143.   
Goodfellow, I. J.; Pouget-Abadie, J.; Mirza, M.; Xu, B.; Warde-Farley, D.; Ozair, S.; Courville, A.; and Bengio, Y. 2014. Generative adversarial nets. Advances in neural information processing systems, 27.   
Huang, C.-y.; Lin, Y. Y.; Lee, H.-y.; and Lee, L.-s. 2021. Defending your voice: Adversarial attack on voice conversion. In 2021 IEEE Spoken Language Technology Workshop (SLT), 552–559. IEEE.   
Li, J.; Ye, D.; Tang, L.; Chen, C.; and Hu, S. 2023. Voice Guard: Protecting Voice Privacy with Strong and Imperceptible Adversarial Perturbation in the Time Domain. In IJ-CAI, 4812–4820.   
Madry, A.; Makelov, A.; Schmidt, L.; Tsipras, D.; and Vladu, A. 2017a. Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083. Madry, A.; Makelov, A.; Schmidt, L.; Tsipras, D.; and Vladu, A. 2017b. Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083. Mittag, G.; and Moller, S. 2021. Deep learning based as- ¨ sessment of synthetic speech naturalness. arXiv preprint arXiv:2104.11673.   
Rix, A. W.; Beerends, J. G.; Hollier, M. P.; and Hekstra, A. P. 2001. Perceptual evaluation of speech quality (PESQ)- a new method for speech quality assessment of telephone networks and codecs. In 2001 IEEE InternationalConference on Acoustics, Speech, and Signal Processing. Proceedings (Cat. No. 01CH37221), volume 2, 749–752. IEEE. Sakoe, H.; and Chiba, S. 1978. Dynamic programming algorithm optimization for spoken word recognition. IEEE transactions on acoustics, speech, and signal processing, 26(1): 43–49.   
Wang, Z.; Bovik, A. C.; Sheikh, H. R.; and Simoncelli, E. P. 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4): 600–612.