# VocalBridge: Latent Diffusion-Bridge Purification for Defeating Perturbation-Based Voiceprint Defenses

Maryam Abbasihafshejani

AHM Nazmus Sakib

Department of Computer Science

University of Texas at San Antonio

San Antonio, Texas

Department of Computer Science

University of Texas at San Antonio

San Antonio, Texas

Murtuza Jadliwala

Department of Computer Science

University of Texas at San Antonio

San Antonio, Texas

Abstract—The rapid progress of speech synthesis technologies, such as text-to-speech (TTS) and voice conversion (VC), has intensified security and privacy concerns surrounding voice cloning. Recent defenses attempt to prevent unauthorized cloning by embedding protective perturbations into speech, aiming to obscure speaker identity while maintaining intelligibility. However, adversaries can employ state-of-the-art purification strategies to remove these perturbations, recover genuine acoustic characteristics, and regenerate cloneable voices. Although such threats are increasingly realistic, the robustness of these protective mechanisms under adaptive purification has not been adequately studied.

Existing purification and denoising approaches, mostly designed for adversarial noise in automatic speech recognition (ASR) or classification, fail to preserve the fine-grained acoustic features that define a speaker’s voice. As a result, they often degrade the perceptual quality and introduce distortions in the speaker-embedding space. Recent diffusion-based purification frameworks provide stronger denoising against perturbations targeting Automatic Speaker Verification (ASV) systems, but many of them operate directly on the waveform or require transcriptdependent phoneme alignment during inference. Although they outperform ASR-oriented denoising methods, these approaches still struggle to fully remove adversarial perturbations, which limits recovery fidelity and reduces overall performance. To address these limitations, we propose Diffusion-Bridge (VocalBridge), a purification model that learns a latent mapping from perturbed to clean speech within the EnCodec latent space. It employs a time-conditioned 1D-UNet denoiser to perform reverse diffusion under a cosine noise schedule, enabling efficient, transcriptfree purification while preserving speaker-discriminative cues. We also introduce a Whisper-Guided Phoneme variant that incorporates lightweight linguistic conditioning from a Whisperbased phoneme alignment module. Unlike prior text-conditioned diffusion models that rely on transcripts or external language prompts, our approach operates entirely in the acoustic domain and requires no linguistic supervision

These findings expose the fragility of current perturbationbased defenses and underscore the need for more resilient safeguards against evolving voice-cloning threats.

Index Terms—purifier, formatting, Diffusion-Bridge, Voice cloning, Perturbation-Based Defenses

## I. INTRODUCTION

Recent advances in generative AI have intensified concerns about security, privacy, and data ownership. A particularly troubling area is speech generation, including text-to-speech (TTS) and voice conversion (VC) models [1]. These technologies can now produce highly realistic audio deepfakes that enable impersonation, misinformation, and identity theft [2]– [4]. For example, scammers recently cloned the voice of a public figure to deceive a family member into believing that they were in legal trouble [5]. Critically, AI-generated fake voices can bypass state-of-the-art ASV, which still face challenges of generalization and robustness [6]. Beyond individual fraud cases, synthetic speech has already been used in large-scale political misinformation campaigns, fraudulent financial transactions, and social engineering attacks targeting enterprises and government agencies. In February 2025, Italian authorities uncovered an AI-voice scam in which criminals impersonated the Italian Defence Minister and used voicecloning technology to trick a prominent entrepreneur into wiring nearly C1 million (≈ \$1.04 million) to a foreign account [7]. In response to these escalating threats, both OpenAI [8] and the U.S. Federal Trade Commission (FTC) [9] have released reports warning about the security, privacy, and societal implications of voice conversion and synthetic voice technologies, highlighting the urgency of developing robust methods to detect and mitigate malicious uses of modern speech generation systems.

To mitigate these risks, researchers have begun exploring proactive voice protection methods that make a user’s speech unlearnable for synthesis models. The key idea is to introduce carefully crafted perturbations or targeted transformations that prevent TTS/VC systems from extracting speaker-specific features (timbre), while keeping the audio natural and intelligible for legitimate use such as communication or authentication by ASV sytems. These perturbations reduce the ability of synthesis models to accurately mimic the identity of a speaker, ensuring that any fake audio generated from protected speech is likely to be of lower quality or unable to pass ASV systems [10], [11].

However, the performance of existing proactive defenses remains underexplored in scenarios where attackers can purify protected speech before using it. Prior purification research has primarily focused on removing adversarial noise targeting automatic speech recognition (ASR) systems, leaving open the question of whether current methods can remove protective perturbations designed to defend against voice cloning and speaker verification attacks. To demonstrate the insufficiency of existing purification approaches and the need for more robust speech protection techniques, we introduce VocalBridge, a comprehensive purification attack that effectively removes protective perturbations and enables attackers to bypass modern speaker verification systems.

VocalBridge is a diffusion-bridge purification model that learns a latent mapping from protected to clean speech representations within a compact encoded domain. The model employs a time-conditioned 1D U-Net denoiser that models the reverse diffusion process inside the EnCodec latent space, trained to estimate residual perturbations under a cosinebased noise schedule. This design enables efficient, transcriptfree purification while preserving the speaker characteristics needed for cloning and verification attacks. Our results show that VocalBridge substantially surpasses existing purification methods in both attack effectiveness and performance

Our contributions can be summarized as follows:

1) Comprehensive evaluation of proactive defenses. We conduct a large-scale evaluation of five state-of-the-art proactive defenses across six modern voice-synthesis tools and three widely used ASV encoders, providing an up-to-date view of how current protection mechanisms behave under diverse synthesis pipelines. Our study incorporates recent synthesis models, multiple defense strategies and purification tools, and a large dataset constructed from both VCTK and LibriSpeech, resulting in a comprehensive assessment of proactive protection in contemporary voice-cloning scenarios.

2) VocalBridge Diffusion-Bridge purification in the speech latent space. We introduce a bridge-diffusion purification model that operates in the speech latent space and learns a reverse mapping from diffused defensive perturbations representations back to their clean latent counterparts. We further enhance performance with a lightweight Whisper-based conditioning mechanism that provides phoneme-level guidance without requiring transcripts. Across our evaluation metrics, Verification Recovery (VR), Mean Opinion Score (MOS), and Word Error Rate (WER), VocalBridge consistently improves authentication restoration, perceptual quality, and intelligibility compared to recent purification methods, while also demonstrating strong generalization and robustness to adaptive protection strategies.

3) Dataset creation for community evaluation. We generate a large dataset that includes clean, protected, and purified speech samples suitable for evaluating proactive defenses, providing a useful benchmark for future research.

## II. BACKGROUND AND RELATED WORK

## A. Deepfake Speech Synthesis Techniques

Deepfake speech synthesis involves the use of AI and machine learning techniques to generate audio clips that replicate the vocal characteristics of a target speaker, including timbre, prosody, and articulation patterns. Deepfake speech synthesis techniques can be broadly classified into: (i) TTS and (ii) VC tools.

TTS tools convert written text into spoken words. Some of the earliest techniques, such as concatenative synthesis, were capable of producing clear speech; however, the result often sounded unnatural and robotic [12], [13]. In contrast to these, modern deep learning approaches can generate remarkably natural and human-like vocal output [14], [15]. These systems typically operate in a two-stage pipeline. First, an acoustic model converts the input text into a low-level acoustic representation, such as a Mel spectrogram, which encodes phonetic information, pitch, and timing. Then, a neural vocoder synthesizes a high-fidelity audio waveform from this representation. The use of deep neural networks in both stages is what allows these models to capture the complex nuances of human speech. A key application of this technology is voice cloning [16], which aims to create a synthetic voice that sounds indistinguishable from a specific target speaker. This is achieved by training or fine-tuning a TTS model on a collection of audio recordings of that individual. By learning from these samples, the model captures the unique vocal characteristics of the target, including their timbre, intonation patterns, and accent [17]–[19]. This allows the trained model to be able to effectively generate a cloned voice.

Unlike TTS, voice conversion or VC modifies a source speaker’s vocal attributes to match those of some target speaker, while preserving the original linguistic content. This is achieved by separating speaker-dependent acoustic features from the phonetic information and re-synthesizing the audio.

## B. Speech Privacy and Protection Mechanisms

Deepfake speech synthesis can seriously hamper user privacy and create security risks, consequently various defensive strategies have been explored in literature. Although anonymizing a speaker’s voice is an effective strategy to protect privacy [20], [21], it is not sufficient on its own as a defense; protection against deepfake audio also requires accurate detection and prevention. Detection focuses mainly on the identification of synthetic (or cloned) audio and includes techniques such as classification using Mel-frequency cepstral coefficient (MFCC) features [22], emotion recognition [23], acoustic signal analysis by simulating auditory effects of the human ear [24] and ASV systems [25]. Preventive defense techniques are more retroactive in nature and aimed at rendering the speech synthesizing capability of audio synthesis tools ineffective. To combat the creation of fake or synthetic speech, several techniques have been proposed that employ adversarial perturbations to protect original audio recordings [26]–[28]. By introducing subtle perturbations or modifications at the data level, these techniques prevent a generative model from successfully cloning the voice during its inference procedure, i.e., the final cloned audio is dissimilar to the desired target audio. Zhang et al. [29] proposed a protection technique called POP which applies imperceptible error-minimizing noises on original speech samples to prevent them from being effectively learned by TTS synthesis models. In their follow-up work, they proposed SPEC [11], which they claim is robust against advanced adaptive adversaries. Dong et al. [30] proposed a Generative Adversarial Network (GAN) based perturbation generation technique to protect against audio deepfake generation. Lastly, techniques such as AntiFake [10] use adversarial examples to disrupt the vocal timbre in synthesized audio, thereby thwarting zero-shot voice cloning and voice conversion by making the output dissimilar to the original speaker.

## C. Adversarial Purification

Because protective perturbations function similarly to adversarial noise added to the input of VC models, an important question is whether existing adversarial purification techniques can remove these perturbations and thereby bypass the protection. Adversarial purification methods operate directly on the input audio and aim to recover a clean signal by eliminating adversarial distortions before the audio reaches the downstream VC system.

Unlike adversarial training, which modifies or retrains the model itself, purification does not require access to or control over the target VC model. This makes purification an appealing strategy in our setting, where the protection mechanism must defend against external VC models without altering them. Moreover, purification methods can generalize to unseen or adaptive perturbations because they focus on restoring the input signal rather than relying on a fixed set of adversarial examples encountered during training.

Recent research shows that generative model-based denoising approaches, especially diffusion models, are highly effective at removing perturbations from audio [31], [32]. The technique by Wu et al. [33], for instance, involves initially adding slight noise to the adversarial audio, then using a DDPM [34] based diffusion model to denoise it and restore the clean audio. Diffpure [35] uses a plug-and-play diffusion module as a pre-precessing step to remove perturbations. Guo et al. [36] build upon this work by building a more robust solution focused on Automated Speech Recognition (ASR), as ASR models often focus on the low frequency components and neglect high frequency components. They accomplish this by using a hierarchical diffusion framework using a pre-trained guided diffusion model. Tan et al. [37] employ a one-shot unconditional Mel spectrogram diffusion model and work on both time and frequency domains to remove perturbations. First, Gaussian perturbations are added into time domain signal through interpolation methods, then the samples’ Mel spectrograms are purified with a diffusion model. While prior works primarily targeted adversarial perturbations added to compromise ASR models, Fan et al. [38] propose a two-step method to remove protective perturbations whose goal is to make speech un-learnable by generative models. In the first step, the perturbed audio is purified using an unconditional pre-trained diffusion model. In the second step, an Ornstein–Uhlenbeck SDE–based refinement model [39], guided by phoneme information, is employed to more closely align the speech audio with the original signal.

In the context of image data, Li et al. [40] investigate the limited robustness of purification methods based on simple pre-trained diffusion models. Specifically, they evaluate DiffPure [35] under gradient-based PGD attacks [41] and demonstrate that both clean accuracy and adversarial robustness are significantly degraded. Although several follow-up efforts [42]–[46] have sought to improve the robustness of DiffPure, their practical applicability is constrained by the high computational cost of Monte Carlo sampling required by these approaches [47]. To address these limitations, Li et al. propose ADBM, an approach that constructs a reverse bridge from the diffused adversarial image data distribution to the clean data distribution. Their method assumes that adversarial noise $\epsilon _ { a }$ is injected at the beginning of the forward diffusion process during training. The model is trained to predict the original noise ϵ together with a scaled version of the adversarial noise $\epsilon _ { a } .$ , enabling the reverse diffusion process to effectively remove adversarial perturbations while denoising the input.

## III. THREAT MODEL AND SECURITY OBJECTIVES

We consider the threat model shown in Fig. 1, where a user seeks to prevent the misuse of their publicly available voice recordings, while an attacker attempts to generate deepfake audio to impersonate the user and deceive ASV systems. Parties and Objectives: The system consists of three components: the user, the adversary, the ASV system.

• User. The user legitimately produces utterances for communication or authentication and may publicly share them (e.g., on social media or online platforms). To prevent misuse, the user applies a defense mechanism $P _ { \psi }$ , parameterized by ψ, to each clean utterance x before release, yielding a protected version:

$$
\begin{array} { r } { x _ { p } = P _ { \psi } ( x ) . } \end{array}
$$

The goal of $P _ { \psi }$ is to prevent malicious voice synthesis while preserving the naturalness and verifiability of the utterance for legitimate ASV use.

• Attacker. The Attacker collects publicly available protected utterances $\{ x _ { p } \}$ and attempts to clone or synthesize utterances that mimic the user’s timbre. The adversary’s objective is to generate a cloned utterance x˜ that is accepted by the ASV system as genuine.

• Verifier (ASV System). The ASV system evaluates a similarity score $s ( \cdot , \cdot )$ between an input utterance and the enrollment utterance $x _ { e }$ of the legitimate user. Verification is successful if $s ( \cdot , \cdot ) \ > \ \tau$ , where τ is the system’s decision threshold.

User Objectives: The defense mechanism $P _ { \psi }$ should ensure that the released protected utterances $\{ x _ { p } \}$ maintain usability while resisting impersonation. Specifically, it should satisfy the following properties:

<!-- image-->  
Fig. 1: This figure illustrates the threat model, demonstrating how an attacker leverages VocalBridge to bypass existing defenses and execute voice-cloning attacks.

• Imperceptibility: Perturbations introduced by $P _ { \psi }$ must be imperceptible to human listeners so that $x _ { p }$ remains natural, intelligible, and high-quality while preserving the linguistic content and intent of x.

• Verifiability: The legitimate user should still be accepted by the ASV system after protection:

$$
s ( x _ { p } , x _ { e } ) \geq \tau ,
$$

where $s ( \cdot , \cdot )$ is the cosine similarity between speaker embeddings, and τ is the ASV decision threshold.

• Inimitability: Synthetic (or deepfake) utterances x˜ generated from the protected samples $x _ { p }$ should not be accepted by ASV as genuine:

$$
s ( \tilde { x } , x _ { e } ) < \tau .
$$

Attacker Objectives: In our setting, the adversary’s goal is to invalidate the user’s protection objectives through purification and synthesis. Specifically, the attacker aims to:

• Restore Verifiability: Purification aims to restore ASV verification by converting protected utterances that are initially rejected $( s ( x _ { p } , x _ { e } ) < \tau )$ into purified utterances that meet the acceptance threshold $( s ( x _ { \mathrm { p u r } } , x _ { e } ) \ge \tau )$

• Preserve Imperceptibility: Ensure that the purified audio remains natural and intelligible to human listeners.

• Break Inimitability: Enable synthetic or cloned speech generated from purified utterances to be accepted by ASV as genuine, thereby demonstrating that the user’s inimitability defense no longer holds.

Together, these objectives quantify the attack’s ability to reverse the user’s protective effects (restoring authentication functionality) while maintaining perceptual quality and facilitating re-cloning.

Attacker Capabilities: The adversary operates under realistic and bounded capabilities:

• The adversary can collect short segments of publicly available protected utterances $\{ x _ { p } \}$ and use them to train or prompt few-shot or zero-shot TTS/VC models.

• To improve synthesis, the adversary may apply a perturbation-removal mechanism $R _ { \phi }$ (parameterized by ϕ) to approximate the clean utterance:

$$
x _ { r } = R _ { \phi } ( x _ { p } ) .
$$

The purified utterance $x _ { r }$ is then input to a generative model $G _ { \theta }$ (parameterized by θ) to produce the imitation $\tilde { x } = G _ { \theta } ( x _ { r } )$

• The adversary can access a small auxiliary dataset of paired samples $\{ ( \boldsymbol { x } ^ { ( i ) } , \boldsymbol { x } _ { p } ^ { ( i ) } ) \} _ { i = 1 } ^ { N }$ from non-target speakers to adapt or train $R _ { \phi }$ , but these pairs do not include the target user.

• The adversary has no white-box access to $P _ { \psi }$ (i.e., no access to its parameters, gradients, or internals) and no knowledge of the internal parameters or thresholds of the ASV system.

## IV. AUDIO DIFFUSION BRIDGE MODEL

To align with the clean data distribution, we propose the Audio Diffusion-Bridge Model, named VocalBridge, a purification mechanism that recovers clean audio from data protected by perturbation-based defenses by constructing a reverse bridge directly from the perturbed data distribution to the clean data distribution.

We build on the Adversarial Diffusion-Bridge Model (ADBM) proposed by Li et al. [40], which was originally developed for image data. Directly applying ADBM to the audio domain is non-trivial due to the much higher temporal resolution, different noise characteristics, and the instability of waveform-level diffusion. To address these challenges, we adapt the ADBM formulation to speech and implement all diffusion operations within the latent space of a neural audio codec (EnCodec).This design allows diffusion to be performed on a more compact representation and enables the model to learn the mapping between protected and clean audio using paired training samples, rather than relying on classifier guidance as in the original ADBM. The decoder subsequently reconstructs the audio from the purified latent representation.

## A. Training Objective

The model is a diffusion-based purification framework designed to suppress protective perturbation in audio latent representations. Its forward process follows the DDPM formulation, except that the initial state is perturbed by a protection noise term $\varepsilon _ { a }$ rather than beginning from the clean latent sample. For each clean latent vector $\mathbf { z } _ { c } ,$ the initial noised state is defined as

$$
z _ { 0 } ^ { a } = \mathbf { z } _ { c } + \varepsilon _ { a } .
$$

The forward diffusion process is expressed as

$$
z _ { t } ^ { a } = \sqrt { \bar { \alpha } _ { t } } z _ { 0 } ^ { a } + \sqrt { 1 - \bar { \alpha } _ { t } } \varepsilon , \qquad \varepsilon \sim \mathcal { N } ( 0 , I ) , \quad 0 \leq t \leq T ,
$$

<!-- image-->  
Fig. 2: VocalBridge Training

where $\bar { \alpha } _ { t }$ denotes the cosine noise schedule and $T$ represents the terminal diffusion step used for purification.

The reverse process learns a sequence $\{ \hat { z } _ { t } \} _ { t : T  0 }$ that maps the diffused, protection-noised latent distribution $\left( z _ { T } ^ { a } \right)$ back to the clean distribution $\left( \mathbf { z } _ { c } \right)$ . To align the intermediate diffusion trajectory, scaling coefficients $c _ { \mathrm { i n } } ( t )$ and $c _ { \mathrm { t g t } } ( t )$ are defined following the derivation in [40]:

$$
c _ { \mathrm { i n } } ( t ) = \frac { \bar { \alpha } _ { T } ( 1 - \bar { \alpha } _ { t } ) } { \sqrt { \bar { \alpha } _ { t } } ( 1 - \bar { \alpha } _ { T } ) } , \qquad c _ { \mathrm { t g t } } ( t ) = \frac { \bar { \alpha } _ { T } \sqrt { 1 - \bar { \alpha } _ { t } } } { ( 1 - \bar { \alpha } _ { T } ) \sqrt { \bar { \alpha } _ { t } } } .
$$

The bridged latent variable and the corresponding effective noise target are then given by

$$
z _ { t } ^ { d } = \sqrt { \bar { \alpha } _ { t } } \mathbf { z } _ { c } + \sqrt { 1 - \bar { \alpha } _ { t } } \varepsilon + c _ { \mathrm { i n } } ( t ) \varepsilon _ { a } ,
$$

$$
\varepsilon _ { \mathrm { e f f } } = \varepsilon + c _ { \mathrm { t g t } } ( t ) \varepsilon _ { a } .\tag{1}
$$

(2)

A neural network $\varepsilon _ { \theta } { \left( z _ { t } ^ { d } , t \right) }$ is trained to estimate the effective noise term $\varepsilon _ { \mathrm { e f f } }$ . The bridge loss is defined as

$$
\mathcal { L } _ { \mathrm { b r i d g e } } = \mathbb { E } _ { t , \varepsilon } \Big [ \| \varepsilon _ { \theta } ( z _ { t } ^ { d } , t ) - \varepsilon _ { \mathrm { e f f } } \| _ { 2 } ^ { 2 } \Big ] .
$$

This formulation is equivalent to the simplified bridge loss of Li et al. [40], up to constant scaling factors omitted for clarity following Ho et al. [34].

To enhance latent-space consistency, an auxiliary $L _ { 1 }$ regularization term is introduced using the reconstructed clean latent estimate:

$$
\hat { z } _ { 0 } = \frac { z _ { t } ^ { d } - \sqrt { 1 - \bar { \alpha } _ { t } } \varepsilon _ { \theta } ( z _ { t } ^ { d } , t ) - \left( c _ { \mathrm { i n } } ( t ) - \sqrt { 1 - \bar { \alpha } _ { t } } c _ { \mathrm { t g t } } ( t ) \right) \varepsilon _ { a } } { \sqrt { \bar { \alpha } _ { t } } } .
$$

The total objective becomes

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { b r i d g e } } + \lambda _ { z _ { 0 } } \| \hat { z } _ { 0 } - \mathbf { z } _ { c } \| _ { 1 } , \qquad \lambda _ { z _ { 0 } } \ll 1 .\tag{3}
$$

As training progresses, the relative contribution of $\varepsilon _ { a }$ decreases with smaller t, enabling the model to progressively suppress protective noise and recover the clean latent representation. Fig. 2 illustrates the overall training process of VocalBridge.

## B. Inference

During inference, a protected waveform is first encoded into the latent domain, producing the protected latent representation ${ \bf z } _ { a }$ . To initiate the reverse diffusion trajectory, we construct a noisy latent state at terminal time $T$ as

$$
z _ { T } = \sqrt { \bar { \alpha } _ { T } } { \bf z } _ { a } + \sqrt { 1 - \bar { \alpha } _ { T } } \varepsilon , \varepsilon \sim { \cal N } ( 0 , I ) .
$$

A multi-step DDIM-style reverse process is then applied to map $z _ { T }  \hat { z } _ { 0 }$ . At each diffusion step t, the denoiser $\hat { \varepsilon } _ { \theta } ( z _ { t } , t )$ predicts the noise present in the current latent, and the clean latent estimate is obtained via

$$
\hat { z } _ { 0 } = \frac { z _ { t } - \sqrt { 1 - \bar { \alpha } _ { t } } \hat { \varepsilon } _ { \theta } ( z _ { t } , t ) } { \sqrt { \bar { \alpha } _ { t } } } .
$$

Iterating this update over a schedule $t = T , t _ { T - 1 } , \dots , 0$ yields a final latent estimate $\hat { z } _ { 0 }$ that approximates the clean latent representation $\mathbf { z } _ { c } .$ The purified waveform is obtained by decoding the recovered latent through the EnCodec decoder.

## C. Whisper-Guided Phoneme Conditioning

We extend our diffusion-bridge purifier with a lightweight phoneme-guided conditioning mechanism that provides additional temporal structure for speech denoising and refinement. A Whisper-based alignment front-end is used to estimate approximate phoneme timings directly from the waveform, enabling the model to better preserve speech content while aligning the purified output with the clean speech distribution. Unlike prior works (Fan et al. [38]), which rely on groundtruth transcripts, our approach operates purely on acoustic inputs and does not require transcript supervision.

Let y denote the phoneme guidance signal, a time-aligned sequence that encodes the phoneme structure of an utterance. This signal is derived from a phoneme alignment map Λ produced by the Whisper-based aligner and combined with a simple acoustic prior to form a smooth guidance track. During training, y is extracted from the clean waveform, standardized and passed through a bounded nonlinearity, then RMSmatched to the bridged latent $z _ { t } ^ { d }$ before channel concatenation:

$$
x _ { \mathrm { i n } } = \left[ z _ { t } ^ { d } \parallel \mathrm { R M S M a t c h } ( y , z _ { t } ^ { d } ; \gamma ) \right] .
$$

The denoising network is thus conditioned on both the noisy latent and the phoneme guidance, enabling the reverse process to remove protective perturbations while preserving speech structure.

At inference time, the phoneme guidance signal y is extracted directly from the input audio using the same Whisperbased alignment module, without access to clean references or transcripts. The resulting guidance acts as a soft temporal prior rather than a strict constraint, making the method robust to moderate alignment errors and residual noise. If phoneme alignment is unavailable, the system automatically falls back to an unconditioned mode by substituting a zero-valued guidance channel, ensuring stable and fully compatible purification.

## D. Network Architecture

The purifier network adopts a 1D U-Net architecture with time-step conditioning. It operates entirely in the EnCodec latent domain, taking as input the bridged latent $z _ { t } ^ { d }$ (and optionally the phoneme guidance track y) and predicting the effective noise term $\hat { \varepsilon } _ { \theta } \big ( z _ { t } ^ { d } , t , y \big )$ . The U-Net employs a hierarchical encoder–decoder structure with residual Time-Delay Neural Network (TDNN) blocks to capture both local and long-range temporal dependencies in the audio representation. Time conditioning is implemented via sinusoidal embeddings and Feature-wise Linear Modulation (FiLM) modulation at each decoder stage, enabling accurate diffusion-step awareness.

In the phoneme-guided variant, an additional input channel concatenates the normalized guidance signal Λ(t) to the latent tensor, allowing linguistic information to influence denoising while preserving the bridge consistency. The network is trained using an AdamW optimizer with cosine learning rate decay and gradient clipping for stability.

## V. EXPERIMENT AND EVALUATION

We evaluate the proposed purification framework in terms of Restoration of Verifiability and perceptual quality, which together aim to break the defense mechanism’s inimitability property by enabling cloning of the protected speech and the generation of high-quality fake speech.

## A. Experimental Setup

All experiments are implemented in PyTorch and conducted on an NVIDIA A100 server (80 GB GPU, 512 GB RAM) for model training and purification. For lighter inference workloads, including TTS and VC synthesis, we additionally utilize NVIDIA Tesla V100 GPUs (32 GB). The remaining experimental configurations are described in the following subsections.

1) Datasets: We conducted our experiments on two widely used benchmark speech datasets: LibriSpeech [48] and VCTK [49]. LibriSpeech contains high-quality audiobook recordings from multiple speakers reading diverse transcripts, while VCTK consists of 110 English speakers with varied accents and recording conditions. For LibriSpeech, we select 40 speakers and use their first-chapter recordings, which provide phonetically balanced sentences suitable for evaluating intelligibility and speaker consistency. For VCTK, we use the first 50 utterances from each speaker. To train and evaluate our proposed purifier VocalBridge, we split the VCTK corpus into 30 speakers for training and 80 speakers for testing, and the LibriSpeech corpus into 27 speakers for training and 13 speakers for testing. All splits are gender-balanced to ensure a fair comparison across datasets. In total, the evaluation uses 4,526 test samples and 13,414 training samples across both datasets.

To construct evaluation data, we apply protective-noise defenses to both datasets and then synthesize cloned audio using selected TTS and VC tools. For TTS cloning, we sample previously unseen sentences from VCTK and assign each as the cloning target for a randomly chosen protected speaker. For VC cloning, we randomly select 50 utterances from source speakers and convert each into the voice of every protected speaker. This process produces approximately 42,289 synthesized samples per defense using TTS tools (209,936 total across all defenses) and 14,438 samples per defense using three VC tools (80,404 total VC samples).

To evaluate VocalBridge Against baseline purification methods, each system is applied to the protected samples before cloning through both TTS and VC pipelines. For each purification method evaluated, we generate 4,526 purified test samples and a total of 290,340 synthesized samples (209,936 from TTS and 80,404 from VC). These synthesized outputs are then used to evaluate Restoration of Verifiability as well as perceptual quality.

2) Protection Tools: As discussed previously, perturbationbased voice defense methods aim to protect speech data by injecting optimized perturbations that make it unlearnable for TTS and VC models. In this work, we select a representative set of state-of-the-art defenses based on two criteria: (1) Effectiveness: each method is explicitly designed to degrade the ability of TTS and VC systems to synthesize realistic voices while preserving perceptual speech quality (quality defense) and preserving speaker identity for ASV (timbre defense). (2) Availability: We require that each defense method provides publicly released model checkpoints along with the corresponding optimization code.Consequently, we included publicly available perturbation-based speech defense methods that demonstrate strong reported performance. Specifically, we chose the following defenses in this work:

• SafeSpeech: SafeSpeech [11] proposes a training-time defense that targets both zero-shot and fine-tuning voice cloning attacks by making audio samples unusable for learning in TTS models. Its central mechanism, Speech PErturbative Concealment (SPEC), uses a surrogate generative model to guide the creation of perturbations that, as the authors claim, ensure there is nothing to learn from the protected data during model training. Rather than relying on inference-time adversarial examples, Safe-Speech introduces perturbations optimized through Melspectrogram loss and KL divergence with noise distributions, aiming to degrade both the speaker identity (timbre) and synthesis quality in any speech generated from the protected audio.

• Attack-VC: Attack-VC [50] defends speeches by leveraging an encoder-decoder structure, where the encoder is divided into a content encoder and a speaker encoder. The content encoder, which captures the linguistic content, is left untouched to preserve what is being said, while the defense specifically targets the speaker encoder that extracts the unique identity of the speaker. The mechanism adds carefully crafted perturbations to the input utterances. These perturbations are designed to alter the output spectrogram, the speaker embedding, or both, so that even when the linguistic content remains clear, voice conversion models cannot accurately clone or identify the speaker’s voice. The changes are minimally perceptible to humans, but significantly reduce the risk of voice imitation or misuse by advanced VC technologies.The embedding attack offers the best tradeoff between effectiveness and speed and is robust across models; accordingly, we use the embedding attack in our evaluation.

• Pivotal Objective Perturbation (POP): POP [29] is a method that adds small imperceptible adversarial perturbations to audio prior to release, with the goal of making the data unlearnable for TTS voice cloning models. The authors claim that POP generates these perturbations by optimizing only the reconstruction loss (the difference between real and synthesized audio) and that, since this loss is shared across nearly all TTS models, the approach is universally effective, efficient, and transferable across architectures. They report that POP generates protected audio that sounds natural to human listeners, but when TTS models are trained on this protected data, the resulting synthetic speech becomes noisy and unusable.

• Anti-Fake: AntiFake [10] targets identity disruption in synthesized speech by perturbing the speaker embedding space, ensuring that generated voices no longer resemble the original speaker to humans or machines. It employs two optimization strategies: a threshold-based method that pushes embeddings beyond a set distance from the original, and a target-based method that moves embeddings toward a different speaker. To ensure transferability to unknown TTS models, it optimizes perturbations using an ensemble of diverse speaker encoders. It also introduces a perceptual loss based on human hearing sensitivity and signal-to-noise ratios to preserve audio quality. Finally, it incorporates a human-in-the-loop process for selecting target voices and validating perceptual dissimilarity.

• Active Defense Against Voice Conversion Through Generative Adversarial Network (GAN-ADV): GAN-ADV [51] introduces an adversarial defense framework that generates perturbations in the Mel-spectrogram domain using a generator-discriminator architecture (GAN), aiming to disrupt VC systems without perceptibly altering audio quality. The system includes a simulation module (SWCSM) that mimics the lossy process of waveform reconstruction and re-extraction of features, improving robustness to real-world inference pipelines. A substitute VC model is used during training to provide gradient signals, and perturbation optimization balances three objectives: fooling a discriminator (GAN loss), disrupting VC output (defense loss), and preserving audio fidelity (quality loss). Inference requires only the trained generator, making it efficient at deployment time.

We selected these public defenses because they demonstrate high performance and collectively cover methods that apply perturbations in different feature spaces and exhibit varying degrees of transferability across synthesis models. SafeSpeech (SPEC) introduces training-time waveform perturbations guided by a surrogate generative model. Attack-VC perturbs spectrogram representations either by directly distorting the output spectrogram (end-to-end), altering the speakerembedding representation, or combining both through a feedback mechanism; among these, the speaker-embedding attack variant is reported as the most effective, which we adopt. POP also operates in the waveform domain but optimizes perturbations with respect to Mel-spectrogram reconstruction loss. AntiFake manipulates speaker embeddings via thresholdand target-based optimization over an ensemble of encoders, enabling partial transferability across synthesis systems. GAN-ADV generates perturbations in the Mel-spectrogram domain through a GAN framework designed to remain robust after waveform reconstruction.

3) Voice Cloning tools: We assume that the adversary has access to a diverse set of state-of-the-art open-source TTS and VC models for synthesizing/cloning voice, each representing distinct architectural paradigms and recent advances in neural speech synthesis. We select six representative models

• VALL-E-X: VALL-E-X [52] is a neural codec language model that predicts discrete acoustic tokens conditioned on source-language speech and target-language text. It supports zero-shot, cross-lingual TTS and speech-to-speech translation while preserving speaker timbre, emotion, and acoustic environment cues.

• Tortoise-TTS: Tortoise-TTS [53] combines an autoregressive transformer-based acoustic model with a diffusion decoder and UnivNet vocoder. It first predicts compressed acoustic tokens and then refines them into expressive highquality waveforms.

• StyleTTS2: StyleTTS2 [54] is a text-to-speech model that employs style diffusion and adversarial training with large speech language models (SLMs) to generate naturalsounding audio. It models speaking style as a latent random variable via diffusion, enabling style-consistent synthesis without requiring reference speech.

• VQMIVC: VQMIVC [55] is a one-shot voice conversion model that leverages vector quantization for content encoding and mutual information maximization for disentangling content, speaker, and pitch representations in an unsupervised manner.

• HierSpeech++: HierSpeech++ [56] is a hierarchical variational autoencoder integrating text-to-semantic-unit modeling, prosody control, and speech super-resolution. It incorporates normalizing flow modules for high-fidelity, zero-shot synthesis and conversion.

• DiffHierVC [57]: A hierarchical VC framework utilizing two diffusion models, DiffPitch and DiffVoice, for sequential conversion. It achieves voice style transfer via a source–filter encoder that disentangles speech representations, with masked priors that enhance speaker adaptation quality.

4) Speaker Verification Systems: We employ three standard ASV systems to evaluate speaker identity preservation and recovery: x-vector, ECAPA, and d-vector, implemented using pretrained models from SpeechBrain [58] and Resemblyzer [59]. Each system maps an input waveform to a fixeddimensional embedding that represents the vocal identity of the speaker. Verification between a test utterance and an enrolled speaker centroid is performed via cosine similarity:

$$
\mathrm { s c o r e } ( \tilde { \mathbf { e } } , \mathbf { c } ) = \frac { \tilde { \mathbf { e } } ^ { \top } { \mathbf { c } } } { \| \tilde { \mathbf { e } } \| _ { 2 } \| \mathbf { c } \| _ { 2 } } ,
$$

where e˜ denotes the test embedding and c is the centroid of the speaker’s enrollment embeddings, computed as the average embedding over the enrollment utterances.

The decision threshold is determined using the equal error rate (EER) criterion, defined as the operating point where the false-accept rate (FAR) equals the false-reject rate (FRR):

$$
\mathrm { E E R } = \mathrm { F A R } ( \tau _ { \mathrm { e e r } } ) = \mathrm { F R R } ( \tau _ { \mathrm { e e r } } ) ,
$$

with $\tau _ { \mathrm { e e r } }$ denoting the equal-error threshold. All ASV embeddings and decision thresholds are computed using our evaluation dataset. To prevent data leakage, the subset used for threshold calibration (the development set) is strictly disjoint from the utterances used for enrollment and purification evaluation. A few clean utterances per speaker are used to compute speaker centroids, while the remaining utterances and their protected or purified counterparts serve as test trials. The ASV systems remain frozen throughout all experiments and function solely as objective evaluators of speaker identity consistency

On the development set which includes 149 speakers from our evaluation dataset, the x-vector, ECAPA, and d-vector systems achieve EERs of 0.0486, 0.006, and 0.0297, respectively, with corresponding thresholds of $\tau _ { \mathrm { e e r } } { = } 0 . 9 5 1$ (x-vector), 0.419 (ECAPA), and 0.750 (d-vector). These calibrated thresholds are fixed for all subsequent purification and recovery experiments.

5) Evaluation Metrics: To assess the effectiveness of VocalBridge, we evaluate it with respect to the attacker objectives defined in Section III.

• Restoration of Verifiability: Following the AntiFake evaluation design [10], which measures defense effectiveness through the Authentication Evasion Reduction Rate (AERR), we adopt the inverse perspective suitable for purification. Our goal is to remove protective perturbations that suppress speaker verification and thus restore authentication. To this end, we introduce the Authentication Restoration Rate (ARR) as the primary metric to quantify the efficacy of purification. ARR measures the proportion of previously unverified (belowthreshold) protected utterances that become successfully verified after purification. For each identity, a clean enrollment centroid c is calculated from a subset of clean enrollment utterances. Let s(e, c) denote the cosine similarity between an embedding e and the enrolled centroid. We denote by $s _ { i } ^ { \mathrm { p r o t } } ~ = ~ \mathrm { s } ( \mathbf { e } _ { i } ^ { \mathrm { \bar { p r o t } } } , \mathbf { c } )$ and $s _ { i } ^ { \mathrm { p u r } } ~ = ~ \mathrm { s } ( \mathbf { e } _ { i } ^ { \mathrm { p u r } } , \mathbf { c } )$ the similarity scores for the protected and purified versions of the same utterance, respectively. Given the equal-error threshold $\tau _ { \mathrm { e e r } }$ in Section V-A4, the ARR is defined as

$$
\mathrm { A R R } ( \tau _ { \mathrm { e e r } } ) = \frac { \sum _ { i } \mathbf { 1 } \{ s _ { i } ^ { \mathrm { p r o t } } < \tau _ { \mathrm { e e r } } ~ \land ~ s _ { i } ^ { \mathrm { p u r } } \geq \tau _ { \mathrm { e e r } } \} } { \sum _ { i } \mathbf { 1 } \{ s _ { i } ^ { \mathrm { p r o t } } < \tau _ { \mathrm { e e r } } \} } ,
$$

Higher ARR values indicate stronger authentication recovery and more effective removal of protective perturbation.

• Imperceptibility: To evaluate perceptual fidelity and naturalness, we use the objective Mean Opinion Score (MOS) predicted by the NISQA model [60]. NISQA estimates speech quality and intelligibility on a scale from 1 (poor) to 5 (excellent), providing an automated approximation of human perceptual judgment. Higher MOS values indicate more natural and intelligible speech, with scores above 3 generally reflecting good audio quality [10].

We also use the Word Error Rate (WER) [61] to assess pronunciation clarity. A pre-trained Whisper-small model [46] is employed for transcription due to its computational efficiency on our large dataset. Higher WER values indicate reduced speech clarity.

6) Purification Baselines: In this section, we provide a brief description of the denoising tools used in our experiments. The set includes adversarial denoisers and protective perturbations removers that represent a diverse range of recent approaches.

• De-Antifake: De-AntiFake [38] is a voice cloning attack evaluation and purification system designed to test and defeat existing speech protection mechanisms that rely on adversarial perturbations. It simulates a realistic attacker who applies purification techniques to remove protective noise from speech before performing voice cloning. The tool introduces a new purification framework called PhonePuRe, which works in two stages; (1) Purification Stage : Uses a diffusion-based model to clean adversarially perturbed audio. (2) Refinement Stage : Employs phoneme-guided alignment to fine-tune the purified speech so it closely matches natural, unprotected speech.

• WavePurifier: WavePurifier [36] is a defensive tool designed to purify audio adversarial examples that target ASR systems. It uses a hierarchical diffusion model that removes perturbations (via forward diffusion) and restores clean speech (via reverse diffusion). The tool divides spectrograms into frequency bands (low, mid, high) and optimizes purification intensity per band, maintaining speech quality while removing attacks. Evaluated on multiple ASR models and attacks, WavePurifier outperforms seven existing defenses, achieving the lowest Character and WERs and the highest purification success rate across diverse scenarios.

<!-- image-->  
Fig. 3: The mean ARR (%) over three speaker verification back-ends (x-vector, ECAPA-TDNN, and d-vector).

• AudioPure: AudioPure [33] is an adversarial purification-based defense pipeline made for acoustic systems using off-the-shelf diffusion models. It uses diffusion models to generate noise, which is added to adversarial audio in a small amount. Then, a reverse sampling step is performed to purify the noisy audio and recover the clean audio. It is a plug-and-play method, which can be applied to any pre-trained classifier without the need for additional fine-tuning or re-training.

• DualPure: DualPure [37] is a real-time purification based defense method against adversarial perturbations. First, it first disrupts the potential malicious perturbations at waveform level in the samples. Following this, an unconditional diffusion model is used to purify the features at the frequency level. Specifically, it first applies a time-domain purifier (TDP) to purify waveform signals, then converts the waveform to a mel spectrogram and applies frequency-domain purification. It achieves good adversarial robustness against both white-box and blackbox attacks.

## B. Experiment Results

1) Restoring Verifiability: VocalBridge vs. Baselines: In this section, we evaluate VocalBridge against four purification mechanisms that an attacker can employ: De-AntiFake (full model including Purification and Refinement), AudioPure, DualPure, and WavePurifier. Among these, De-AntiFake is the only related work explicitly designed to defeat voicecloning defenses by removing their protective perturbations; to the best of our knowledge, no other purification-based attack has been proposed for this setting. The remaining purifiers were originally developed to mitigate adversarial attacks on

ASR systems and are not tailored to our protection-removal threat model. Nevertheless, we include them as baselines to determine whether these purification techniques can inadvertently act as effective protection-removal attacks. Table I and Table II report the ARR of speech synthesized by selected TTS and VC models from purified datasets across the three ASV systems: x-vector, ECAPA-TDNN, and d-vector. As summarized in Fig. 3, VocalBridge-W achieves the strongest overall restoration across protections. Under the Attack-VC protection, VocalBridge-W reaches 45.0% ARR, improving over the best existing method (DualPure at 37.4%) by a margin of 7.6. On the GAN-ADV and POP protections, VocalBridge-W attains 32.8% and 35.6%, exceeding the strongest baselines by 18.1 and 14.0, respectively. On SafeSpeech, VocalBridge-W also provides the highest restoration (28.9%), slightly surpassing prior work. VocalBridge exhibits similar gains, raising Attack-VC performance from 37.4% to 42.1%, GAN-ADV from 14.7% to 23.5%, and POP from 21.6% to 28.2%. The only setting where our models do not lead is AntiFake, where the specialized De-AntiFake method remains higher (39.4% vs. 31.7% for VocalBridge-W). Notably, De-AntiFake requires full access to the AntiFake detector and its groundtruth transcript to operate its refiner, a requirement rarely satisfied in practice, whereas VocalBridge and VocalBridge-W function without any privileged model access, making them substantially more practical and broadly applicable.

Fig. 4 shows the spectrograms of the input and output samples for voice cloning. The visual comparison indicates that VocalBridge-W preserves the structural and spectral details of the clean audio more effectively than the state-of-the-art De-AntiFake baseline. A notable limitation of De-AntiFake is its reliance on clean transcripts, an assumption that often breaks under noisy or distorted conditions such as SafeSpeech and AntiFake. In contrast, VocalBridge and VocalBridge-W require no ground truth transcripts and therefore maintain effectiveness even when transcripts are degraded or unavailable, providing a more practical basis for robustness evaluation.

Overall, these results indicate that VocalBridge and VocalBridge-W provide more reliable purification than existing methods, effectively removing protective perturbations and restoring speaker identity across different protection mechanisms, ASV backends, and synthesis architectures.

2) Evaluation of Cross-Perturbation Generalization of VocalBridge: To assess the generalization capability of our purification framework, we use a Mono model that is trained on only a single perturbation pattern. We train Mono using only the perturbation pattern produced under GAN-ADV and then apply it to audio perturbed by other protections, without any form of adaptation. This setup allows us to evaluate whether learning from just one perturbation type is sufficient for removing a wide range of unseen perturbations. Table III shows that the Mono variant remains closely aligned with the Adaptive model across all protection types.

Overall, these results demonstrate that the VocalBridge maintains high authentication-restoration performance even on perturbation patterns it never encountered during training, indicating strong robustness and generalization across heterogeneous purification defenses.

TABLE I: Authentication Restoration Rate (%) for selected TTS models.
<table><tr><td rowspan="2">Protection</td><td rowspan="2">TTS models</td><td colspan="3">De-AntiFake</td><td colspan="3">DualPure</td><td colspan="3">WavePurifier</td><td colspan="3">AudioPure</td><td colspan="3">VocalBridge</td><td colspan="3">VocalBridge-W</td></tr><tr><td></td><td>xvec ecapa dvec</td><td></td><td>xvec ecapa dvec</td><td></td><td></td><td></td><td></td><td>xvec ecapa dvec</td><td></td><td>xvec ecapa dvec</td><td></td><td></td><td>xvec ecapa dvec</td><td></td><td></td><td>xvec ecapa dvec</td><td></td></tr><tr><td>AntiFake</td><td>StyleTTS2</td><td>86.61 40.34 54.05</td><td></td><td></td><td>13.110.40</td><td></td><td>0.94</td><td></td><td>52.028.72</td><td>1.40</td><td>19.97</td><td>1.54</td><td>1.65</td><td></td><td>58.17 21.742.76</td><td></td><td>55.27</td><td>12.71</td><td>7.24</td></tr><tr><td></td><td>Tortoise-TTS</td><td>62.05 53.75 59.90</td><td></td><td></td><td>19.57</td><td>1.11</td><td>0.98</td><td>24.37</td><td>9.66</td><td>0.84</td><td></td><td>25.564.55</td><td>2.07</td><td></td><td>47.8635.868</td><td>8.37</td><td></td><td>42.8330.3411.57</td><td></td></tr><tr><td></td><td>VALL-E-X</td><td>45.4226.20 25.36</td><td></td><td></td><td>63.72</td><td>7.94</td><td>1.12</td><td></td><td>25.796.61</td><td>0.37</td><td></td><td>59.413.44</td><td>1.94</td><td></td><td>38.1314.6811.54</td><td></td><td>48.91</td><td>16.4316.44</td><td></td></tr><tr><td>Avg (TTS)</td><td></td><td>64.69</td><td>40.10</td><td>46.44</td><td>32.133.15</td><td></td><td>1.01</td><td>34.06</td><td>8.33</td><td>0.87</td><td>34.98</td><td>3.18</td><td>1.89</td><td>48.05 24.09</td><td></td><td>7.56</td><td>49.00</td><td>19.83</td><td>11.75</td></tr><tr><td>Attack-VC StyleTTS2</td><td></td><td>32.32</td><td>10.12</td><td>9.07</td><td>48.54</td><td>36.93</td><td>54.47</td><td>5.29</td><td>14.30</td><td>10.05</td><td>45.27</td><td>29.51</td><td>47.32</td><td>61.21</td><td>60.03</td><td>67.23</td><td></td><td>63.8660.66 64.37</td><td></td></tr><tr><td></td><td>Tortoise-TTS</td><td>32.52</td><td>9.60</td><td>3.94</td><td>36.56</td><td>23.89</td><td>28.29</td><td>3.45</td><td>16.26</td><td>2.53</td><td>41.74</td><td>19.90</td><td>20.52</td><td>38.07</td><td>25.91</td><td>15.67</td><td>38.30 24.40</td><td></td><td>10.03</td></tr><tr><td>Avg (TTS)</td><td>VALL-E-X</td><td>29.47</td><td>8.00</td><td>3.36</td><td>73.03</td><td>52.84</td><td>25.25</td><td>15.63</td><td>12.24</td><td>1.93</td><td>69.51</td><td>42.71</td><td>19.02</td><td>39.67</td><td>36.02</td><td>19.79</td><td>41.48 35.51</td><td></td><td>24.97</td></tr><tr><td>GAN-ADV</td><td></td><td>31.44</td><td>9.24</td><td>5.46</td><td>52.71</td><td>37.89</td><td>36.00</td><td>8.12</td><td>14.27</td><td>4.84</td><td>52.18</td><td>30.71</td><td>28.95</td><td>46.32</td><td></td><td>40.66 34.23</td><td>47.88 40.19</td><td></td><td>33.12</td></tr><tr><td></td><td>StyleTTS2</td><td>39.82</td><td>14.93</td><td>6.09</td><td>17.37</td><td>2.44</td><td>4.05</td><td>6.24</td><td>7.56</td><td>1.87</td><td>14.41</td><td>2.43</td><td>5.04</td><td>49.01</td><td></td><td>36.92 31.31</td><td></td><td>57.5840.24 34.22</td><td></td></tr><tr><td></td><td>Tortoise-TTS</td><td></td><td>32.7610.88</td><td>2.64</td><td></td><td>28.055.99</td><td>3.11</td><td>3.71</td><td>8.19</td><td>0.68</td><td>22.99</td><td>7.26</td><td>2.62</td><td></td><td></td><td>38.9916.799.90</td><td></td><td>49.1112.047.25</td><td></td></tr><tr><td></td><td>VALL-E-X</td><td>29.808.92</td><td></td><td>3.27</td><td>58.56</td><td>8.68</td><td>4.73</td><td>12.94</td><td>4.67</td><td>0.95</td><td>61.95</td><td>5.94</td><td>4.69</td><td>40.8619.98 17.71</td><td></td><td></td><td></td><td>47.9920.8318.45</td><td></td></tr><tr><td>Avg (TTS)</td><td></td><td></td><td>34.13 11.58</td><td>4.00</td><td>34.66</td><td>5.70</td><td>3.96</td><td>7.63</td><td>6.81</td><td>1.17</td><td>33.12</td><td>5.21</td><td>4.12</td><td>42.95 24.56</td><td></td><td>19.64</td><td></td><td>51.56 24.37 19.97</td><td></td></tr><tr><td>POP</td><td>StyleTTS2</td><td></td><td></td><td>38.9130.21 43.38</td><td>14.24</td><td>0.00</td><td>0.00</td><td></td><td>10.8214.77</td><td>5.82</td><td></td><td>12.350.09</td><td>0.00</td><td>40.0050.04 58.76</td><td></td><td></td><td></td><td>45.49 52.78 60.50</td><td></td></tr><tr><td></td><td>Tortoise-TTS</td><td></td><td>33.7517.92</td><td>12.81</td><td>28.10</td><td>2.66</td><td>2.44</td><td>2.69</td><td>7.39</td><td>0.43</td><td>22.02</td><td>1.25</td><td>1.30</td><td>42.8615.0911.75</td><td></td><td></td><td></td><td>45.0618.04 10.94</td><td></td></tr><tr><td></td><td>VALL-E-X</td><td>31.02 20.98</td><td></td><td>17.77</td><td>59.92</td><td>7.59</td><td>7.19</td><td>8.49</td><td>10.12</td><td>1.11</td><td>54.92</td><td>6.59</td><td>5.11</td><td>35.16 31.03 40.87</td><td></td><td></td><td></td><td>35.26 29.25 43.35</td><td></td></tr><tr><td>Avg (TTS) SafeSpeech StyleTTS2</td><td></td><td></td><td></td><td>34.56 23.04 24.66</td><td></td><td>34.093.42</td><td>3.21</td><td>7.33</td><td>10.76</td><td>52.45</td><td>29.762.64</td><td></td><td>2.14</td><td>39.34 32.05 37.13</td><td></td><td></td><td>41.94 33.36 38.27</td><td></td><td></td></tr><tr><td></td><td></td><td>58.7521.67</td><td></td><td>17.89</td><td></td><td>33.482.04</td><td>0.90</td><td>20.28</td><td>7.85</td><td>1.97</td><td>33.72</td><td>1.56</td><td>2.09</td><td>54.87 21.13 31.01</td><td></td><td></td><td></td><td>56.4422.92 31.06</td><td></td></tr><tr><td></td><td>Tortoise-TTS</td><td></td><td>49.3020.54</td><td>13.38</td><td></td><td>34.665.04</td><td>3.84</td><td>14.21</td><td>7.79</td><td>1.04</td><td>28.83</td><td>7.49</td><td>4.00</td><td>39.445.88</td><td></td><td>5.25</td><td>33.91</td><td>4.90</td><td>5.78</td></tr><tr><td></td><td>VALL-E-X</td><td></td><td>35.29 21.50</td><td>7.65</td><td>62.49</td><td>3.55</td><td>3.93</td><td>10.97</td><td>3.53</td><td>0.22</td><td>63.71</td><td>4.28</td><td>1.98</td><td></td><td>41.6515.33</td><td>16.18</td><td>37.0312.67</td><td></td><td>17.42</td></tr><tr><td>Avg (TTS)</td><td></td><td>47.78 21.24</td><td></td><td>12.98</td><td></td><td>43.543.54</td><td>2.89</td><td>15.15</td><td>6.39</td><td>1.08</td><td>42.09</td><td>4.44</td><td>2.69</td><td>45.32</td><td>14.11</td><td>17.48</td><td>42.46 13.50 18.09</td><td></td><td></td></tr></table>

TABLE II: Authentication Restoration Rate (%) for selected VC models.
<table><tr><td rowspan="2">Protection</td><td rowspan="2">VC models</td><td colspan="3">De-AntiFake</td><td colspan="3">DualPure</td><td colspan="3">WavePurifier xvec ecapa dvec</td><td colspan="3">AudioPure</td><td colspan="3">VocalBridge</td><td colspan="3">VocalBridge-W</td></tr><tr><td>xvec ecapa dvec</td><td></td><td></td><td></td><td>xvec ecapa dvec</td><td></td><td></td><td></td><td></td><td></td><td></td><td>xvec ecapa dvec</td><td></td><td></td><td>xvec ecapa dvec</td><td></td><td>xvec ecapa dvec</td><td></td></tr><tr><td rowspan="2">AntiFake</td><td>DifHierVC</td><td>77.3923.9974.05</td><td></td><td></td><td>8.70</td><td>1.54</td><td>1.42</td><td></td><td>40.692.85</td><td>4.82</td><td>18.12</td><td>1.54</td><td>1.42</td><td>37.055.57</td><td></td><td>13.42</td><td>26.81</td><td>3.73</td><td>9.96</td></tr><tr><td>HierSpeechpp</td><td>2.75</td><td>2.89</td><td>16.75</td><td>1.61</td><td>8.6431.28</td><td></td><td>0.00</td><td>6.94</td><td>17.77</td><td></td><td></td><td>10.92 29.28 37.63</td><td>2.40</td><td>14.9345.69</td><td></td><td></td><td>35.5451.9375.63</td><td></td></tr><tr><td rowspan="2">Avg (VC)</td><td>vQMIVC</td><td>6.32</td><td>17.84 33.03</td><td></td><td>0.00</td><td>0.34</td><td>0.00</td><td>0.24</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>67.6537.97</td><td></td><td>5.83</td><td>72.3043.73</td><td></td><td>8.52</td></tr><tr><td></td><td>28.82</td><td>14.91</td><td>41.27</td><td>3.44</td><td>3.51</td><td>10.90</td><td>13.64</td><td>3.26</td><td>7.20</td><td>9.01</td><td>10.27</td><td>13.68</td><td>35.70</td><td>19.49 21.65</td><td></td><td></td><td>44.88 33.13 31.37</td><td></td></tr><tr><td rowspan="2">Attack-VC DiffHierVC</td><td></td><td>13.79</td><td>2.28</td><td>11.34</td><td></td><td>13.7936.48</td><td>50.85</td><td>46.67</td><td></td><td>13.0822.03</td><td>22.58</td><td>18.77</td><td>43.38</td><td>30.0054.18</td><td></td><td>73.45</td><td></td><td>34.38 53.21</td><td>70.34</td></tr><tr><td>HierSpeechpp</td><td>1.49</td><td>0.78</td><td>8.15</td><td>3.36</td><td>20.10</td><td>33.03</td><td>0.00</td><td>5.75</td><td>10.70</td><td>6.29</td><td>30.60</td><td>）41.28</td><td>4.67</td><td></td><td>24.7542.51</td><td></td><td>19.72 41.63</td><td>56.88</td></tr><tr><td rowspan="3">Avg (VC)</td><td>vQMIVC</td><td></td><td>48.8535.59 27.93</td><td></td><td></td><td>38.0441.70</td><td>55.68</td><td>33.36</td><td>7.98</td><td>2.70</td><td>35.59</td><td>39.00</td><td>38.92</td><td>58.66 48.21</td><td></td><td>58.38</td><td>59.29 51.29</td><td></td><td>58.92</td></tr><tr><td></td><td>21.38</td><td>12.88</td><td>15.81</td><td></td><td>18.40 32.76</td><td>46.52</td><td>26.68</td><td>8.94</td><td>11.81</td><td>21.49</td><td>29.46</td><td>541.19</td><td>31.11</td><td>42.38</td><td>58.11</td><td>37.80</td><td>48.71</td><td>62.05</td></tr><tr><td>DiffHierVC</td><td>25.91</td><td>3.60</td><td>2.66</td><td>13.64</td><td>1.46</td><td>3.98</td><td>8.65</td><td>2.57</td><td>4.31</td><td>13.84</td><td>1.51</td><td>5.27</td><td>22.07</td><td></td><td>11.9525.27</td><td>35.14</td><td>17.43</td><td>38.34</td></tr><tr><td rowspan="3">GAN-ADV</td><td>HierSpeechpp</td><td>17.83</td><td>2.77</td><td>2.22</td><td>3.55</td><td>0.82</td><td>2.77</td><td>1.40</td><td>1.21</td><td>4.10</td><td>9.66</td><td>1.35</td><td>2.36</td><td>10.24</td><td>7.99</td><td>17.09</td><td></td><td></td><td>33.8014.5329.59</td></tr><tr><td>VQMIVC</td><td>20.06 25.19</td><td></td><td>15.86</td><td>23.33</td><td>8.39</td><td>15.65</td><td>14.39</td><td>2.11</td><td>2.04</td><td>24.74</td><td>6.21</td><td>6.80</td><td>24.67</td><td>25.00 17.69</td><td></td><td></td><td></td><td>59.3943.17 30.61</td></tr><tr><td></td><td>21.27</td><td>10.52</td><td>6.91</td><td>13.51</td><td>3.56</td><td>7.47</td><td>8.15</td><td>1.96</td><td>3.48</td><td>16.08</td><td>3.02</td><td>4.81</td><td>19.00</td><td></td><td>14.98 20.02</td><td></td><td></td><td>42.78 25.04 32.85</td></tr><tr><td rowspan="3">Avg (VC) POP</td><td>DifHierVC</td><td>6.25</td><td>5.09</td><td>36.13</td><td>5.26</td><td>0.47</td><td>0.27</td><td>60.00</td><td>9.29</td><td>2.73</td><td>0.00</td><td>0.69</td><td>0.27</td><td>10.00 22.38 28.42</td><td></td><td></td><td></td><td></td><td>14.2925.93 36.61</td></tr><tr><td>HierSpeechpp</td><td>0.83</td><td></td><td>2.0029.86</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.74</td><td>0.54</td><td>1.10</td><td>0.79</td><td>0.27</td><td>0.00</td><td>0.76</td><td>7.26</td><td>17.63</td><td></td><td></td><td>29.0138.07 49.59</td></tr><tr><td>VQMIVC</td><td></td><td>21.4332.17</td><td>7.94</td><td></td><td>27.52 15.3819.30</td><td></td><td></td><td>10.895.26</td><td>0.00</td><td></td><td>21.4314.29</td><td>7.94</td><td></td><td></td><td>37.61 29.33 28.07</td><td></td><td></td><td>38.7435.0633.33</td></tr><tr><td rowspan="3">Avg (VC) SafeSpeech DiffHierVC</td><td></td><td>9.50</td><td></td><td>13.09 24.64</td><td>10.93</td><td>5.28</td><td>6.52</td><td></td><td>23.885.03</td><td>1.28</td><td>7.41</td><td>5.08</td><td>2.74</td><td>16.13</td><td>19.66 24.71</td><td></td><td></td><td></td><td>27.35 33.02 39.84</td></tr><tr><td></td><td>46.138.81</td><td></td><td>24.97</td><td>23.69</td><td>0.84</td><td>0.62</td><td></td><td>31.563.33</td><td>9.38</td><td>19.57</td><td>1.82</td><td>4.20</td><td>36.96</td><td>5.14</td><td>27.34</td><td>39.146.71</td><td></td><td>34.94</td></tr><tr><td>HierSpeechpp</td><td>61.44</td><td>11.95</td><td>38.10</td><td>23.65</td><td>1.40</td><td>1.40</td><td></td><td>34.233.97</td><td>11.92</td><td>15.24</td><td>2.50</td><td>5.63</td><td>44.76</td><td>8.97</td><td>36.21</td><td></td><td>46.1510.44 39.28</td><td></td></tr><tr><td>Avg (VC)</td><td>VQMIVC</td><td>29.62</td><td>15.0929.47</td><td>45.7311.95 30.85</td><td>26.39 24.58</td><td>11.84 16.10 4.69</td><td>6.04</td><td>23.24 29.68</td><td>7.56 4.95</td><td>12.68 11.33</td><td>30.89 21.90</td><td>11.52 5.28</td><td>13.62 7.82</td><td>63.01 48.24</td><td>32.47 15.53 26.71</td><td>16.59</td><td>66.9535.94 50.75</td><td></td><td>19.02 17.70 31.08</td></tr></table>

## C. Effectiveness of Whisper-Guided Phoneme Conditioning

Integrating Whisper features into VocalBridge (forming VocalBridge-W in Tables I and II) improves purification stability and helps the model preserve speaker-relevant phonetic structure. Across all protection settings, Whisper-guided conditioning provides consistent gains over our simple Vocal-Bridge purifier. For instance, on GAN-ADV, VocalBridge-W improves encoder-averaged ARR of VC models from 19/15/20% to 43/25/33%, and on POP, ARR rises from 16/20/25% to 27/33/40%.

1) Evaluation of Speech Quality and Imperceptibility: Table IV reports the average NISQA-TTS MOS across all VC/TTS systems. The protected speech retains perceptual quality close to the original (3.36 vs. 3.57). Existing purification defenses exhibit substantially lower quality, with average MOS values ranging from 2.95 to 3.27. Our method matches the quality of the protected samples (3.36) while outperforming all prior purification approaches. Fig. 6 reports the average WER for different protection mechanisms on synthetic speech. VocalBridge attains the lowest WER (0.258), outperforming all competing approaches.

These results show that VocalBridge preserves high perceptual fidelity and avoids the over-smoothing and spectral distortions present in baseline methods, allowing attackerused VC/TTS models to replicate speaker characteristics more accurately.

<!-- image-->  
Fig. 4: The spectrogram comparison shows that VocalBridge-W better maintains the structure and detail of the clean samples than the leading baseline De-AntiFake.

<!-- image-->  
Fig. 5: ARR under different adaptive strategies.

## D. Subjective Evaluation of Speech Quality

We conducted a user study with 27 participants to evaluate the subjective quality of the audio generated by our system. The study consisted of a demographic questionnaire followed by a set of listening questions in which participants rated the perceptual quality of synthesized audio created by cloning purified samples with VocalBridge. Participants rated each audio clip using a four-point scale: Good, Average, Poor, or Terrible. The survey included audio obtained by purifying perturbations introduced by Antifake, Attack-VC, GAN-ADV, POP, and SafeSpeech, and subsequently cloning the purified audio with Tortoise-TTS; each participant was presented with a random subset of these samples. One attention-check question containing obvious white-noise corruption was included to ensure participant reliability. As shown in Fig. 5, 75.2% of all ratings fell within the Good, Average, or Excellent categories, with only a small fraction marked as Poor or Terrible, indicating that the purified-and-cloned audio is generally perceived as natural and intelligible. The survey was created using Qualtrics and deployed through Amazon Mechanical Turk.

## E. Adaptive Protection

We also evaluate an adaptive protection scenario in which the protection mechanism has white-box access to our purification model, including its gradients, and can therefore optimize its perturbations accordingly. We adopt the adaptive-attack methodology used in De-Antifake [38]. Because the overall purification function, from input waveform to purified output, is effectively non-differentiable due to components such as EnCodec quantization and stochastic diffusion sampling, we apply their Backward Pass Differentiable Approximation (BPDA) strategy. BPDA treats the purifier as an identity mapping during backpropagation, yielding surrogate gradients that enable end-to-end perturbation optimization despite these non-differentiable operations. As in De-Antifake, we also use Expectation Over Transformation (EOT) to account for randomness in the diffusion process. We average gradient estimates over 1, 5, 10, and 15 stochastic runs, which provide a stable gradient approximation under stochastic sampling.

TABLE III: Authentication Restoration Rate (%) for VC and TTS under Adaptive and Mono purification models
<table><tr><td rowspan="2">Protection</td><td rowspan="2">Model</td><td>VocalBridge</td><td>VocalBridge</td><td>(Mono)</td></tr><tr><td>xvec ecapa dvec</td><td>xvec ecapa</td><td>dvec</td></tr><tr><td rowspan="5">AntiFake</td><td>DiffHierVC HierSpeechpp</td><td>37.05 5.57 13.42</td><td>18.12 2.62</td><td>2.87</td></tr><tr><td></td><td>2.40 14.93 45.69</td><td>2.40 13.72</td><td>32.49</td></tr><tr><td>VQMIVC</td><td>67.65 37.97 5.83</td><td>66.90 36.61</td><td>3.59</td></tr><tr><td>StyleTTS2</td><td>58.17 21.74 2.76</td><td>33.13 4.83</td><td>2.15</td></tr><tr><td>Tortoise-TTS</td><td>47.86 35.86 8.37</td><td>38.78 6.43</td><td>2.22</td></tr><tr><td>Avg (All)</td><td>VALL-E-X</td><td>38.13 14.68 11.54 41.88 21.79 14.60</td><td>34.00 12.46 32.22 12.78</td><td>12.04 9.23</td></tr><tr><td rowspan="6">POP</td><td>DiffHierVC</td><td>10.00 22.38 28.42</td><td>15.00 28.47</td><td>30.60</td></tr><tr><td>HierSpeechpp</td><td>0.76 7.26 17.63</td><td>3.05 8.99</td><td>12.95</td></tr><tr><td>VQMIVC</td><td>37.61 29.33 28.07</td><td>42.48 332.47</td><td>31.58</td></tr><tr><td>StyleTTS2</td><td>40.00 50.04 58.76</td><td>49.17 49.68</td><td>53.05</td></tr><tr><td>Tortoise-TTS</td><td></td><td>42.33 13.45</td><td>7.69</td></tr><tr><td>VALL-E-X</td><td>42.86 15.09 11.75 35.16 31.03 40.87</td><td>34.86 32.07</td><td>37.61</td></tr><tr><td>Avg (All)</td><td></td><td>27.73 25.86 30.92</td><td>31.15 27.52</td><td>28.91</td></tr><tr><td rowspan="6">Attack-VC</td><td>DiffHierVC</td><td>30.00 54.18 73.45</td><td>30.00 54.55</td><td>63.84</td></tr><tr><td>HierSpeechpp</td><td>4.6724.75 42.51</td><td>4.732 24.21</td><td>31.50</td></tr><tr><td>vQMIVC</td><td>58.66 48.21 58.38</td><td>57.84 55.98</td><td>52.43</td></tr><tr><td>StyleTTS2</td><td>61.21 60.03 67.23</td><td>61.31 59.97</td><td>63.85</td></tr><tr><td>Tortoise-TTS</td><td>38.07 25.91 15.67</td><td>38.61 26.49</td><td>13.34</td></tr><tr><td>VALL-E-X</td><td>39.67 36.02 19.79</td><td>39.82 35.48</td><td>19.43</td></tr><tr><td>Avg (All)</td><td></td><td>38.71 41.52 46.17</td><td>38.72</td><td>42.78 40.73</td></tr><tr><td rowspan="6">SafeSpeech</td><td>DiffHierVC</td><td>36.96 5.14 27.34</td><td>28.96</td><td>4.14</td><td>14.48</td></tr><tr><td>HierSpeechpp</td><td>44.76 8.97 36.21</td><td>44.33</td><td>7.22</td><td>15.87</td></tr><tr><td>VQMIVC</td><td>63.01 32.47</td><td>16.59 57.42</td><td>30.77</td><td>10.73</td></tr><tr><td>StyleTTS2</td><td>54.87 21.13</td><td>31.01 47.08</td><td>16.98</td><td>20.29</td></tr><tr><td>Tortoise-TTS</td><td>39.44 5.88</td><td>5.25 28.63</td><td>4.44</td><td>2.45</td></tr><tr><td>VALL-E-X</td><td>41.65 15.33</td><td>16.18 35.82</td><td>11.47</td><td>9.54</td></tr><tr><td></td><td></td><td></td><td>40.37 12.50</td><td>12.23</td></tr><tr><td>Avg (All)</td><td></td><td>46.78 14.82 22.10</td><td></td><td></td></tr></table>

<!-- image-->  
Fig. 6: Average WER of protection mechanisms on synthetic voices.

To evaluate the effectiveness of our method, we first purify the speech that has been adaptively protected, using VocalBridge-W, and then clone the resulting purified audio with StyleTTS2. We measure the attack success using ARR and Speaker Verification Accuracy (SVA) with the x-vector model, where SVA is defined as the fraction of cloned samples accepted as genuine by the ASV system, computed as the average binary match decision across all cloned samples. As shown in Fig. 7 and Fig. 8, under these adaptive strategy settings, ASV remains above 75% and ARR stays above 20%. These results indicate that, even with white-box knowledge, developing effective adaptive protections against our purification method is still difficult, highlighting the risks we have identified.

<!-- image-->  
Fig. 7: SVA under different adaptive strategies.

<!-- image-->  
Fig. 8: ARR under different adaptive strategies.

## VI. DISCUSSION AND LIMITATIONS

## A. Phoneme-Guided Refinement

Our method uses the Whisper small model to generate phoneme-level alignments for Λ-guided refinement. This choice was made because of limited computational resources, and it reduces alignment accuracy, especially when the audio is noisy or adversarial. As a result, the phoneme guidance provides only small improvements. The resource limitations also prevented us from experimenting with larger or more robust alignment models. Future work could explore stronger Whisper variants to obtain more reliable Λ features and greater denoising benefits.

TABLE IV: Mean Opinion Score (MOS) predicted by NISQA-TTS (1–5, higher is better).
<table><tr><td>VC/TTS</td><td>Original</td><td>Protected</td><td>De-AntiFake</td><td>DualPure</td><td>WavePurifier</td><td>AudioPure</td><td>VocalBridge(Ours)</td></tr><tr><td>StyleTTS2</td><td>3.68</td><td>3.61</td><td>3.13</td><td>2.88</td><td>3.31</td><td>2.86</td><td>3.59</td></tr><tr><td>Tortoise-TTS</td><td>3.92</td><td>3.58</td><td>3.44</td><td>3.34</td><td>2.40</td><td>3.23</td><td>3.62</td></tr><tr><td>VALL-E-X</td><td>3.59</td><td>3.27</td><td>3.05</td><td>3.20</td><td>2.74</td><td>3.19</td><td>3.29</td></tr><tr><td>DiffHierVC</td><td>3.43</td><td>3.28</td><td>3.47</td><td>2.93</td><td>3.25</td><td>2.86</td><td>3.24</td></tr><tr><td>HierSpeechpp</td><td>3.88</td><td>3.63</td><td>3.81</td><td>3.41</td><td>3.54</td><td>3.23</td><td>3.59</td></tr><tr><td>VQMIVC</td><td>2.91</td><td>2.78</td><td>2.73</td><td>2.82</td><td>2.46</td><td>2.80</td><td>2.81</td></tr><tr><td> Avg (All)</td><td>3.57</td><td>3.36</td><td>3.27</td><td>3.10</td><td>2.95</td><td>3.03</td><td>3.36</td></tr></table>

## B. Ethics Consideration

The rapid advancement of speech synthesis technologies has ensued a race between voice cloning techniques and protective countermeasures. By highlighting the flaws in the existing perturbation-based countermeasures, this study adds to this adversarial dynamic. We recognize the work’s multifaceted implications; while the primary motivation is to assess the robustness of existing safeguards and thus motivate the development of more resilient solutions, the proposed Diffusion-Bridge purification model also serves as an effective method for circumventing those same safeguards. We firmly believe that the potential benefits of enhancing safeguards against unauthorized speech synthesis and voice cloning far outweigh the risks of misuse of our findings, particularly as we intend to not make our source code publicly accessible. In the interest of open research and advancing science, we will grant access to the source code upon request and only after a thorough review of the request and confirmation of its intent. To ensure that the developers of the evaluated protection tools can appropriately respond and make necessary adaptations, we plan to disclose our findings to them upon acceptance of this manuscript and definitely prior to the publication of this work.

The results of our work highlight an urgent need for the community to rethink perturbation-based approaches and explore fundamentally new strategies for safeguarding voice data. Future protection mechanisms must be designed with robustness to advanced pre-processing and purification in mind, ensuring they remain effective as adversarial capabilities continue to advance.

## VII. CONCLUSION

This paper examined the vulnerability of contemporary voice-protection mechanisms to purification-based attacks and introduced VocalBridge, a bridged latent-diffusion model designed for reconstructing speaker identity. VocalBridge achieves effective identity recovery across both TTS and VC pipelines, and its Whisper-guided extension, VocalBridge-W, further stabilizes phonetic structure and improves reconstruction quality further.

Our evaluations show that VocalBridge and VocalBridge-W substantially outperform existing purification approaches. We additionally assess the generalization capabilities of our model, demonstrating that attackers do not require explicit knowledge of a protection system’s perturbation pattern to mount successful reconstruction attacks. Our adaptive-protection study shows that even with complete white-box access, creating robust defenses against our purification method is still challenging, emphasizing the severity of the risks we expose.

Overall, our results reveal that current speech-protection techniques remain vulnerable to latent-space diffusion–based purification attacks. By releasing our evaluation framework and bridged-diffusion models, we aim to support the development of stronger and more principled defenses for speech privacy and synthetic-media authentication.

## REFERENCES

[1] D. M. Ballesteros, Y. Rodriguez-Ortega, D. Renza, and G. Arce, “Deep4snet: Deep learning for fake speech classification,” Expert Systems with Applications, vol. 184, p. 115465, 2021. [Online]. Available: https://doi.org/10.1016/j.eswa.2021.115465

[2] D. Nici. (2024, February) FCC Outlaws Use of AI-Faked Voices in Robocalls. Accessed: 2025-05-13. [Online]. Available: https://www.fo rbes.com/advisor/personal-finance/fcc-bans-ai-voices-in-robocalls/

[3] N. Y. Times. (2024, April) Welcome to scam world. Accessed: 2025-05-13. [Online]. Available: https://www.nytimes.com/2024/04/21 /style/scams-identity-theft.html

[4] P. Semansky. (2024, January) Fake biden robocall. Accessed: 2025-03- 27. [Online]. Available: https://www.nbcnews.com/tech/misinformation/ joe-biden-newhampshire-robocall-fake-voice-deep-ai-primary-rcna135 120

[5] A. Mitchell. (2024, October) car crash ai scam. Accessed: 2025-05-13. [Online]. Available: https://nypost.com/2024/10/08/tech/

[6] E. Jamdar and A. K. Belman, “Syntheticpop: Attacking speaker verification systems with synthetic voicepops,” https://arxiv.org/abs/2502.09553, February 2025, arXiv preprint arXiv:2502.09553.

[7] S. Angius and R. Staff, “Italian police freeze cash in ai voice scam,” Reuters, Feb. 2025, accessed: 2025-11-13. [Online]. Available: https: //www.reuters.com/technology/artificial-intelligence/italian-police-freez e-cash-ai-voice-scam-that-targeted-business-leaders-2025-02-12/

[8] OpenAI, “Navigating the challenges and opportunities of synthetic voices,” 2024, accessed: 2025-11-13. [Online]. Available: https: //openai.com/index/navigating-the-challenges-and-opportunities-of-syn thetic-voices/

[9] Federal Trade Commission, “Ftc submits comment to fcc on work to protect consumers from potential harmful effects of AI,” 2024, accessed: 2025-11-13. [Online]. Available: https://www.ftc.gov/news-e vents/news/press-releases/2024/07/ftc-submits-comment-fcc-work-pro tect-consumers-potential-harmful-effects-ai

[10] Z. Yu, S. Zhai, and N. Zhang, “Antifake: Using adversarial audio to prevent unauthorized speech synthesis,” in Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security, ser. CCS ’23. New York, NY, USA: Association for Computing Machinery, 2023, p. 460–474. [Online]. Available: https: //doi.org/10.1145/3576915.3623209

[11] Z. Zhang, D. Wang, Q. Yang, P. Huang, J. Pu, Y. Cao, K. Ye, J. Hao, and Y. Yang, “Safespeech: Robust and universal voice protection against malicious speech synthesis,” 2025. [Online]. Available: https://arxiv.org/abs/2504.09839

[12] Y. Tabet and M. Boughazi, “Speech synthesis techniques. a survey,” International Workshop on Systems, Signal Processing and their Applications, WOSSPA, pp. 67–70, 2011. [Online]. Available: https://api.semanticscholar.org/CorpusID:18115403

[13] B. Boashash, Signal Processing and Their Applications, Tipaza, Algeria, 9, 2011.

[14] Z. Almutairi and H. Elgibreen, “A review of modern audio deepfake detection methods: Challenges and future directions,” Algorithms, vol. 15, no. 5, 2022. [Online]. Available: https: //www.mdpi.com/1999-4893/15/5/155

[15] M. Mcuba, A. Singh, R. A. Ikuesan, and H. Venter, “The effect of deep learning methods on deepfake audio detection for digital investigation,” Procedia Computer Science, vol. 219, pp. 211–219, 2023, cENTERIS – International Conference on ENTERprise Information Systems / ProjMAN – International Conference on Project MANagement / HCist – International Conference on Health and Social Care Information Systems and Technologies 2022. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S1877050923002910

[16] J. Yamagishi, C. Veaux, and K. MacDonald, “Cstr vctk corpus: English multi-speaker corpus for cstr voice cloning toolkit (version 0.92),” 2019. [Online]. Available: https://api.semanticscholar.org/CorpusID: 213060286

[17] fishaudio, “Bert-vits2,” https://github.com/fishaudio/Bert-VITS2, 2024.

[18] M. Kawamura, Y. Shirahata, R. Yamamoto, and K. Tachibana, “Lightweight and high-fidelity end-to-end text-to-speech with multiband generation and inverse short-time fourier transform,” in ICASSP 2023 - IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023.

[19] J. Kim, J. Kong, and J. Son, “Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech,” in Proceedings of the 38th International Conference on Machine Learning (ICML). PMLR, 2021.

[20] F. Fang, X. Wang, J. Yamagishi, I. Echizen, M. Todisco, N. Evans, and J.-F. Bonastre, “Speaker anonymization using x-vector and neural waveform models,” in Proceedings of the 10th ISCA Workshop on Speech Synthesis (SSW 10), 2019, pp. 155–160.

[21] J. Huang, C. Zhang, Y. Ren, Z. Jiang, Z. Ye, J. Liu, J. He, X. Yin, and Z. Zhao, “Mullivc: Multi-lingual voice conversion with cycle consistency,” arXiv preprint, 2024.

[22] I.-C. Yoo, K. Lee, S. Leem, H. Oh, B. Ko, and D. Yook, “Speaker anonymization for personal information protection using voice conversion techniques,” IEEE Access, vol. 8, pp. 198 637–198 645, 2020.

[23] A. Hamza, A. R. R. Javed, F. Iqbal, N. Kryvinska, A. S. Almadhor, Z. Jalil, and R. Borghol, “Deepfake audio detection via mfcc features using machine learning,” IEEE Access, vol. 10, pp. 134 018–134 028, 2022.

[24] E. Conti, D. Salvi, C. Borrelli, B. Hosler, P. Bestagini, F. Antonacci, A. Sarti, M. C. Stamm, and S. Tubaro, “Deepfake speech detection through emotion recognition: A semantic approach,” in Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022, pp. 8962–8966.

[25] L. Blue, K. Warren, H. Abdullah, C. Gibson, L. Vargas, J. O’Dell, K. Butler, and P. Traynor, “Who are you (i really wanna know)? detecting audio deepfakes through vocal tract reconstruction,” in 31st USENIX Security Symposium (USENIX Security 22). USENIX Association, 2022, pp. 2691–2708.

[26] Y. Wang, R. J. Skerry-Ryan, D. Stanton, Y. Wu, R. J. Weiss, N. Jaitly, Z. Yang, Y. Xiao, Z. Chen, S. Bengio et al., “Tacotron: Towards endto-end speech synthesis,” arXiv preprint, 2017.

[27] Z. Yu, S. Zhai, and N. Zhang, “Antifake: Using adversarial audio to prevent unauthorized speech synthesis,” in Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security (CCS). ACM, 2023, pp. 460–474.

[28] C. yu Huang, Y. Y. Lin, H. yi Lee, and L. shan Lee, “Defending your voice: Adversarial attack on voice conversion,” in 2021 IEEE Spoken Language Technology Workshop (SLT). IEEE, 2021, pp. 552–559.

[29] Z. Zhang, Q. Yang, D. Wang, P. Huang, Y. Cao, K. Ye, and J. Hao, “Mitigating unauthorized speech synthesis for voice protection,” in Proceedings of the 1st ACM Workshop on Large AI Systems and Models with Privacy and Safety Analysis, ser. CCS ’24. ACM, Nov. 2023, p. 13–24. [Online]. Available: http://dx.doi.org/10.1145/3689217.3690615

[30] S. Dong, B. Chen, K. Ma, and G. Zhao, “Active defense against voice conversion through generative adversarial network,” IEEE Signal Processing Letters, vol. 31, pp. 706–710, 2024.

[31] Z. Kong, W. Ping, J. Huang, K. Zhao, and B. Catanzaro, “Diffwave: A versatile diffusion model for audio synthesis,” 2021. [Online]. Available: https://arxiv.org/abs/2009.09761

[32] N. Chen, Y. Zhang, H. Zen, R. J. Weiss, M. Norouzi, and W. Chan, “Wavegrad: Estimating gradients for waveform generation,” 2020. [Online]. Available: https://arxiv.org/abs/2009.00713

[33] S. Wu, J. Wang, W. Ping, W. Nie, and C. Xiao, “Defending against adversarial audio via diffusion model,” 2023. [Online]. Available: https://arxiv.org/abs/2303.01507

[34] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” ArXiv, vol. abs/2006.11239, 2020. [Online]. Available: https://api.semanticscholar.org/CorpusID:219955663

[35] W. Nie, B. Guo, Y. Huang, C. Xiao, A. Vahdat, and A. Anandkumar, “Diffusion models for adversarial purification,” in International Conference on Machine Learning, 2022. [Online]. Available: https: //api.semanticscholar.org/CorpusID:248811081

[36] H. Guo, G. Wang, B. Chen, Y. Wang, X. Zhang, X. Chen, Q. Yan, and L. Xiao, “Wavepurifier: Purifying audio adversarial examples via hierarchical diffusion models,” Proceedings of the 30th Annual International Conference on Mobile Computing and Networking, 2024. [Online]. Available: https://api.semanticscholar.org/CorpusID: 274090145

[37] H. Tan, X. Liu, H. Zhang, J. Zhang, Y. Qian, and Z. Gu, “Dualpure: An efficient adversarial purification method for speech command recognition,” in INTERSPEECH, 2024. [Online]. Available: https://doi.org/10.21437/Interspeech.2024-855

[38] W. Fan, K. Chen, C. Liu, W. Zhang, and N. H. Yu, “Deantifake: Rethinking the protective perturbations against voice cloning attacks,” ArXiv, vol. abs/2507.02606, 2025. [Online]. Available: https://api.semanticscholar.org/CorpusID:280129534

[39] A. Sokol, “Intervention in ornstein-uhlenbeck sdes,” 2013. [Online]. Available: https://arxiv.org/abs/1308.2152

[40] X. Li, W. Sun, H. Chen, Q. Li, Y. He, J. Shi, and X. Hu, “Adbm: Adversarial diffusion bridge model for reliable adversarial...” Oct 2024. [Online]. Available: https://openreview.net/forum?id=g0rnZeBguq&nes ting=2&sort=date-desc

[41] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, “Towards deep learning models resistant to adversarial attacks,” ArXiv, vol. abs/1706.06083, 2017. [Online]. Available: https://api.semanticsc holar.org/CorpusID:3488815

[42] J. Wang, Z. Lyu, D. Lin, B. Dai, and H. Fu, “Guided diffusion model for adversarial purification,” ArXiv, vol. abs/2205.14969, 2022. [Online]. Available: https://api.semanticscholar.org/CorpusID:249192338

[43] B. Zhang, W. Luo, and Z. Zhang, “Purify++: Improving diffusionpurification with advanced diffusion models and control of randomness,” ArXiv, vol. abs/2310.18762, 2023. [Online]. Available: https://api.sema nticscholar.org/CorpusID:264590450

[44] ——, “Enhancing adversarial robustness via score-based optimization,” ArXiv, vol. abs/2307.04333, 2023. [Online]. Available: https://api.sema nticscholar.org/CorpusID:259501305

[45] N. Carlini, F. Tramer, K. D. Dvijotham, and J. Z. Kolter, “(certified!!) \` adversarial robustness for free!” ArXiv, vol. abs/2206.10550, 2022. [Online]. Available: https://api.semanticscholar.org/CorpusID:24988964 1

[46] C. Xiao, Z. Chen, K. Jin, J. Wang, W. Nie, M. Liu, A. Anandkumar, B. Li, and D. X. Song, “Densepure: Understanding diffusion models towards adversarial robustness,” ArXiv, vol. abs/2211.00322, 2022. [Online]. Available: https://api.semanticscholar.org/CorpusID: 253244468

[47] J. M. Cohen, E. Rosenfeld, and J. Z. Kolter, “Certified adversarial robustness via randomized smoothing,” ArXiv, vol. abs/1902.02918, 2019. [Online]. Available: https://api.semanticscholar.org/CorpusID: 59842968

[48] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “Librispeech: An asr corpus based on public domain audio books,” in 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2015, pp. 5206–5210.

[49] J. Yamagishi, C. Veaux, and K. MacDonald, “CSTR VCTK corpus: English multi-speaker corpus for CSTR voice cloning toolkit (version 0.92),” 2019.

[50] C.-y. Huang, Y. Lin, H.-y. Lee, and L.-S. Lee, “Defending your voice: Adversarial attack on voice conversion,” 01 2021, pp. 552–559.

[51] C. Xiao, B. Li, J.-Y. Zhu, W. He, M. Liu, and D. Song, “Generating adversarial examples with adversarial networks,” 2019. [Online]. Available: https://arxiv.org/abs/1801.02610

[52] “GitHub - Plachtaa/VALL-E-X: An open source implementation of Microsoft’s VALL-E X zero-shot TTS model. Demo is available in https://plachtaa.github.io/vallex/ — github.com,” https://github.com/P lachtaa/VALL-E-X, [Accessed 14-11-2025].

[53] “GitHub - neonbjb/tortoise-tts: A multi-voice TTS system trained with an emphasis on quality — github.com,” https://github.com/neonbjb/tor toise-tts, [Accessed 14-11-2025].

[54] Y. A. Li, C. Han, V. S. Raghavan, G. Mischler, and N. Mesgarani, “Styletts 2: Towards human-level text-to-speech through style diffusion and adversarial training with large speech language models,” 2023. [Online]. Available: https://arxiv.org/abs/2306.07691

[55] D. Wang, L. Deng, Y. T. Yeung, X. Chen, X. Liu, and H. Meng, “Vqmivc: Vector quantization and mutual information-based unsupervised speech representation disentanglement for one-shot voice conversion,” 2021. [Online]. Available: https://arxiv.org/abs/2106.10132

[56] S.-H. Lee, H.-Y. Choi, S.-B. Kim, and S.-W. Lee, “Hierspeech++: Bridging the gap between semantic and acoustic representation of speech by hierarchical variational inference for zero-shot speech synthesis,” 2023. [Online]. Available: https://arxiv.org/abs/2311.12454

[57] H.-Y. Choi, S.-H. Lee, and S.-W. Lee, “Diff-hiervc: Diffusion-based hierarchical voice conversion with robust pitch generation and masked prior for zero-shot speaker adaptation,” ArXiv, vol. abs/2311.04693, 2023. [Online]. Available: https://api.semanticscholar.org/CorpusID: 260920394

[58] M. Ravanelli, T. Parcollet, P. W. V. Plantinga, A. Rouhe, S. Cornell, L. Lugosch, C. Subakan, N. Dawalatabad, A. Heba, J. Zhong, J.-C. Chou, S.-L. Yeh, S.-W. Fu, C.-F. Liao, E. Rastorgueva, F. Grondin, W. Aris, H. Na, Y. Gao, R. D. Mori, and Y. Bengio, “Speechbrain: A general-purpose speech toolkit,” ArXiv, vol. abs/2106.04624, 2021. [Online]. Available: https://api.semanticscholar.org/CorpusID: 235377273

[59] “GitHub - resemble-ai/Resemblyzer: A python package to analyze and compare voices with deep learning — github.com,” https://github.com/r esemble-ai/Resemblyzer, [Accessed 14-11-2025].

[60] G. Mittag, B. Naderi, A. Chehadi, and S. Moller, “Nisqa: A ¨ deep cnn-self-attention model for multidimensional speech quality prediction with crowdsourced datasets,” Aug. 2021. [Online]. Available: http://dx.doi.org/10.21437/Interspeech.2021-299

[61] Z. Jiang, J. Liu, Y. Ren, J. He, Z. Ye, S. Ji, Q. Yang, C. Zhang, P. Wei, C. Wang et al., “Mega-tts 2: Boosting prompting mechanisms for zeroshot speech synthesis,” arXiv preprint arXiv:2307.07218, 2023.