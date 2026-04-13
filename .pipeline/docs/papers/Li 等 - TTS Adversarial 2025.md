Thinking... Li 等 - 2025 - Adversarial Attacks and Robust Defenses in Speaker Embedding based Zero-Shot Text-to-Speech System.pdf (flash)
# Adversarial Attacks and Robust Defenses in Speaker Embedding based Zero-Shot Text-to-Speech System

Ze Li1,2, Yao Shi3, Yunfei Xu3, Ming Li1,2∗

1 School of Computer Science, Wuhan University, Wuhan, China

2 Suzhou Municipal Key Laboratory of Multimodal Intelligent Systems, Digital Innovation Research Center,

Duke Kunshan University, Kunshan, China

3 AI Center, OPPO, Beijing, China

ming.li369@dukekunshan.edu.cn

Abstract—Speaker embedding based zero-shot Text-to-Speech (TTS) systems enable high-quality speech synthesis for unseen speakers using minimal data. However, these systems are vulnerable to adversarial attacks, where an attacker introduces imperceptible perturbations to the original speaker’s audio waveform, leading to synthesized speech sounds like another person. This vulnerability poses significant security risks, including speaker identity spoofing and unauthorized voice manipulation. This paper investigates two primary defense strategies to address these threats: adversarial training and adversarial purification. Adversarial training enhances the model’s robustness by integrating adversarial examples during the training process, thereby improving resistance to such attacks. Adversarial purification, on the other hand, employs diffusion probabilistic models to revert adversarially perturbed audio to its clean form. Experimental results demonstrate that these defense mechanisms can significantly reduce the impact of adversarial perturbations, enhancing the security and reliability of speaker embedding based zero-shot TTS systems in adversarial environments.

Index Terms—zero-shot text-to-speech, adversarial attack, anti-spoofing, adversarial training, diffusion probabilistic model

## I. INTRODUCTION

With the rapid advancement of deep learning technologies, Text-to-Speech (TTS) systems have made significant progress [1]–[3], particularly with the emergence of Zero-Shot TTS. This technology enables the generation of natural speech for any speaker from short audio samples. Currently, the mainstream Zero-Shot TTS approaches include speaker embeddingbased methods [4]–[6], the large language model (LLM) based ones [7], [8] and their combinations. As shown in Fig.1, speaker embedding based Zero-Shot TTS utilizes a speaker encoder alongside a TTS component, while the large speech generation models formulate the Zero-Shot TTS task as a language modeling task within the neural codec domain.

Although Zero-Shot TTS technology has shown great potential, it also faces new challenges, particularly regarding security and robustness. Malicious attacks have become a severe concern as these systems are increasingly deployed in scenarios that demand high security and reliability. In particular, speaker embedding based Zero-Shot TTS systems are vulnerable to various spoofing attacks. Research works have shown that even with the deep neural networks based approaches [9]–[11], speaker encoders are susceptible to malicious spoofing attacks such as impersonation [12], replay attacks [13], voice conversion [14], and adversarial attacks [15]. Attackers can manipulate their own voice to alter the extracted speaker embeddings, leading the system to generate speech resembling the target speaker, thereby enabling a range of fraudulent activities, including voice forgery and identity impersonation.

<!-- image-->  
(b) LLM-based speaker embedding based zero-shot TTS system.  
Fig. 1. Architecture of traditional and LLM-based speaker embedding based zero-shot TTS systems.

This study focuses on the adversarial attacks in speaker embedding based Zero-Shot TTS systems. Adversarial attacks are typically carried out by generating adversarial examples, which are crafted by introducing imperceptible perturbations through optimization methods such as Fast Gradient-Sign Method [16], Projected Gradient Descent (PGD) [17], optimizer-based [18], and Carlini-Wagner [19] to lead a misclassification.

The prevailing method for defending against adversarial attacks is adversarial training [20], [21], which enhances the model’s robustness by exposing it to adversarial examples during the training phase. Although adversarial training is widely regarded as the most effective defense strategy, it requires substantial computational resources, and the model remains susceptible to unseen attacks that differ from the adversarial methods used during training. Another approach is adversarial purification, which focuses on designing effective purification models to mitigate the adversarial perturbations in input samples. Currently, diffusion models have proven to be the state-of-the-art purification models in both the vision domain [22] and the audio domain, such as in background noise removal [23], speaker verification [24] and speech command recognition [25] tasks.

<!-- image-->  
Fig. 2. Attack and defense framework for speaker embedding based zero-shot TTS system.

This paper investigates adversarial attacks and defenses for speaker embedding based Zero-Shot TTS systems. We experiment with both traditional and LLM-based speaker embedding based zero-shot TTS systems. For the attack phase, we utilize PGD [17] and Adam [26] optimizer-based approaches to generate adversarial examples targeting the speaker encoder of the Zero-Shot TTS system in a white-box attack scenario. For the defense phase, we evaluate and compare two strategies: an active defense through adversarial training and a passive defense via adversarial purification using diffusion models. The Demo Page can be found here1.

## II. METHODS

This section presents our methods for adversarial attacks on speaker embedding based zero-shot TTS system, along with corresponding defense measures, including adversarial training and adversarial purification. The overall framework is illustrated in Fig. 2.

## A. Adversarial Example Generation

Adversarial examples refer to instances with imperceptible perturbations that are deliberately introduced. These perturbations are obtained by solving an optimization problem and lead a well-trained model to make incorrect predictions. This study utilizes PGD [17] and Adam [26] optimizer-based methods to generate adversarial examples. Both methods are gradientbased white-box attacks. We aim to attack the speaker encoder of a speaker embedding based Zero-Shot TTS system by adding small perturbations to the input speech, causing the output speaker embeddings to change, which leads the TTS system to generate speech that mimics the target speaker.

The core idea of the PGD method is to iteratively update the input samples based on the gradient sign of the loss function while constraining the perturbation magnitude. Specifically, given input speeches x, target labels $y ^ { \prime }$ and a well-trained speaker encoder model $f ( \cdot )$ . The predicted labels yˆ can be obtained by computing the index of the maximum cosine similarity between the speaker embeddings of the source speeches and the spoofed ones with adversarial examples, which are extracted by the speaker encoder. The adversarial perturbations δ can be generated by:

$$
\boldsymbol { \hat { y } } = \left[ \underset { \boldsymbol { j } } { \arg \operatorname* { m a x } } c o s i n e ( f ( \boldsymbol { x } ) _ { i } , f ( \boldsymbol { x } + \boldsymbol { \delta } ) _ { j } ) \right] _ { i }\tag{1}
$$

$$
\begin{array} { r l } & { \delta _ { t + 1 } = \delta _ { t } - \alpha \cdot s i g n ( \nabla _ { \delta _ { t } } L o s s ( \hat { y } , y ^ { \prime } ) ) , } \\ & { \qquad \mathrm { s . t . } \quad \| \delta \| _ { \infty } \leq \epsilon } \end{array}\tag{2}
$$

where $s i g n ( \cdot )$ represents the sign of the gradient, α and ϵ control the magnitude of each update and the maximum allowable perturbation, respectively.

Compared to PGD, Adam is an adaptive learning rate optimization algorithm. In each iteration, Adam not only utilizes the current gradient information but also incorporates momentum from previous iterations to update the perturbation:

$$
\begin{array} { r l r } & { } & { m _ { t } = \beta _ { 1 } m _ { t - 1 } + ( 1 - \beta _ { 1 } ) \nabla _ { \delta _ { t } } L o s s ( \hat { y } , y ^ { \prime } ) , \quad } \\ & { } & { v _ { t } = \beta _ { 2 } v _ { t - 1 } + ( 1 - \beta _ { 2 } ) ( \nabla _ { \delta _ { t } } L o s s ( \hat { y } , y ^ { \prime } ) ) ^ { 2 } , } \\ & { } & { \delta _ { t + 1 } = \delta _ { t } - l r \cdot \frac { m _ { t } } { \sqrt { v _ { t } } + \xi } , \quad \mathrm { s . t . } \quad \| \delta \| _ { \infty } \leq \epsilon } \end{array}\tag{3}
$$

where $m _ { t }$ and $v _ { t }$ are the second-moment and second-moment estimates of the gradient, respectively. lr is the learning rate, ξ is the numerical stability constant, $\beta _ { 1 }$ and $\beta _ { 2 }$ are the decay rates for the first and second-moment estimates, respectively.

## B. Adversarial training

Adversarial training is one of the most widely used and effective methods for defending against adversarial attacks, as it enhances the model’s robustness by incorporating adversarial examples into the training process. In this work, we also employ adversarial training. For each speech sample within a batch $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { b }$ , we randomly assign a target speaker label $y ^ { \prime }$ that differs from the source speaker and then apply adversarial attack methods to generate adversarial examples $\{ ( \hat { x } _ { i } , y _ { i } ^ { \prime } ) \} _ { i = 1 } ^ { b }$ . Subsequently, these adversarial examples are labeled with the source speaker’s label and are used alongside the source speech $\{ ( \hat { x } _ { i } , y _ { i } ) \cup ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { b }$ to fine-tune the welltrained speaker encoder model. Finally, the fine-tuned speaker encoder model is used to retrain the zero-shot TTS system.

## C. Adversarial Purification

Diffusion-based adversarial purification is an emerging defense technique against adversarial attacks, which utilizes diffusion models to remove adversarial perturbations from input data, thereby restoring clean speech for effective defense. As a plug-and-play module, diffusion models effectively circumvent the issues of domain shifts and secondary training associated with adversarial training. Furthermore, they do not require training on predefined adversarial examples, which endows them with solid generalization capabilities and allows them to address a wide range of attack methods.

A diffusion model normally consists of a forward diffusion process and a reverse sampling process. The forward diffusion process gradually adds Gaussian noise to the input speech until the distribution of the noisy speech converges to a standard Gaussian distribution:

$$
q ( x _ { t } \mid x _ { 0 } ) = \mathcal { N } ( x _ { t } ; \sqrt { \bar { \alpha } _ { t } } x _ { 0 } , ( 1 - \bar { \alpha } _ { t } ) \mathbf { I } )\tag{4}
$$

where $x _ { 0 }$ is the clean speech, $x _ { t }$ represents the noisy speech at time step t, hyperparameter $\bar { \alpha } _ { t }$ controls the noise level.

The reverse sampling process takes the standard Gaussian noise as input and gradually denoises the noisy speech to recover clean speech. The reverse process is approximated by learning a model $p _ { \theta } ( x _ { t - 1 } \mid x _ { t } )$

$$
p _ { \theta } ( x _ { t - 1 } \mid x _ { t } ) = \mathcal { N } ( x _ { t - 1 } ; \mu _ { \theta } ( x _ { t } , t ) , \Sigma _ { \theta } ( x _ { t } , t ) )\tag{5}
$$

where $\mu _ { \theta } ( x _ { t } , t )$ and $\Sigma _ { \theta } ( x _ { t } , t )$ are represent the predicted mean and covariance for time step t, respectively.

The optimization objective is to minimize the speech reconstruction error, which is achieved using a Mean Squared Error (MSE) loss function:

$$
L = M S E ( x _ { 0 } , \hat { x _ { 0 } } ) = \| x - \hat { x _ { 0 } } \| _ { 2 } ^ { 2 } / N\tag{6}
$$

where $\hat { x _ { 0 } }$ represents the speech obtained by denoising the noisy speech $x _ { t } ,$ , N is the number of samples in speech $x _ { 0 }$

Additionally, considering that adversarial purification might affect clean audio, we introduce a binary classifier before the diffusion module to distinguish between audio samples with adversarial perturbations and those that are clean.

## III. EXPERIMENTAL SETTINGS

## A. Speaker Embedding Baesd Zero-Shot TTS Training

1) Speaker Encoder Training: We utilize the ResNet [27] architecture as the speaker encoder model, including the

ResNet34-based and ResNet101-based ones. The residual block channels are set to {64,128,256,512}, and the output feature maps are aggregated with a global statistics pooling layer that calculates each feature map’s means and standard deviations. The acoustic features are 80-dimensional log Melfilterbank energies with a frame length of 25ms and a hop size of 10ms.

The speaker encoder model is pretrained on the VoxCeleb2 [28] development set and tested on the VoxCeleb1-O [29] test set. We adopt the on-the-fly data augmentation [30] to add additive background noise or convolutional reverberation noise for the time-domain waveform. The MUSAN [31] and RIR Noise [32] datasets are used as noise sources and room impulse response functions, respectively. The speed perturbation [33], which speeds up or down each utterance by a factor of 0.9 or 1.1, is applied to yield shifted pitch utterances that are considered from new speakers. The input utterances are truncated to 2 seconds. We employ the ArcFace [34] classifier, with the margin and scale parameters set as 0.2 and 32, respectively. Network parameters are updated using an SGD optimizer with an initial learning rate of 0.1. The learning rate is decayed by a factor of 0.1 every 10 epochs until 1e-5.

2) Traditional Zero-Shot TTS system Training: We utilize the VITS [3] structure as the traditional zero-shot TTS component. The speaker encoder is the ResNet34-based one. We use the clean subsets of the train and development sets from LibriTTS [35] to train the zero-shot TTS component. The speaker embeddings for the utterances are obtained from the well-trained speaker encoder. The TTS network parameters are updated using the AdamW [36] optimizer with a learning rate 2e-4. The batch size is 32, and the total epoch is 40.

3) LLM-based Zero-Shot TTS system Training: The LLMbased zero-shot TTS system is built on the LauraGPT model [37]. The speaker encoder is the ResNet101-based one. For audio codec, we utilize a pre-trained open-source codec model from the Funcodec toolkit [38], and the text tokenizer is sourced from Qwen [39]. We train the system using the WenetSpeech4TTS [40] training set and HQ-Conversations [41] dataset. The training process consists of two stages. In the first stage, we pre-trained the system using the Premium subset of WenetSpeech4TTS, which includes 945 hours of speech data. The learning rate is set to 1e-3, with 10,000 warm-up steps, a batch size of 160, and a total of 50 epochs. In the second stage, we fine-tuned the system on the 100-hour HQ-Conversations dataset, setting the learning rate to 1e-4, the batch size to 160 and completing 360 epochs.

## B. Speaker Encoder Adversarial Training

The adversarial attack methods are described in II-A. For the PGD method, the perturbation limit ϵ is set to the 5% of the maximum amplitude of each audio sample, with 20 iterations and the step size α is decreased from 4e-3 to 4e-4 with cosine delay. For the Adam optimizer-based method, ϵ is set to the 5% of the maximum amplitude of each audio sample, with 50 iterations and the learning rate lr that decays from 1e-3 to 1e-5 with cosine delay. To balance training costs, we adopt a relatively large perturbation range to accelerate adversarial sample generation. However, one can reduce the upper bound of ϵ and increase the number of iterations to obtain perturbations that are more imperceptible.

TABLE I  
THE PERFORMANCE OF SPEAKER EMBEDDING BASED ZERO-SHOT TTS SYSTEMS UNDER VARIOUS DEFENSE MODES AGAINST DIFFERENT ATTACK METHODS. ORI., TGT., ADV., AND ADV.(SYN) REPRESENT THE SOURCE SPEECH, TARGET SPEECH, ADVERSARIAL SAMPLES, AND THE SPEECH SYNTHESIZED BY THE TTS SYSTEM FROM THE ADVERSARIAL SAMPLES, RESPECTIVELY. MODEL A AND B REPRESENT THE TRADITIONAL AND LLM-BASED SPEAKER EMBEDDING BASED ZERO-SHOT TTS SYSTEMS, RESPECTIVELY.
<table><tr><td>Model</td><td>Defense</td><td>Attack Method</td><td>Attack Success Defense Success Rate[%]</td><td>Rate[%]</td><td>Ori. vs Adv. Tgt. vs Adv. Similarity</td><td>Similarity</td><td>EER[%]</td><td>Ori. vs Adv.(Syn) Tgt. vs Adv.(Syn) Similarity</td><td>Similarity</td></tr><tr><td rowspan="9">A</td><td rowspan="3">None</td><td>None</td><td></td><td></td><td></td><td></td><td>0.957</td><td>0.370</td><td>-0.001</td></tr><tr><td>Adam-based</td><td>99.53</td><td>0.47</td><td>0.134</td><td>0.934</td><td>0.957</td><td>0.048</td><td>0.291</td></tr><tr><td>PGD</td><td>100</td><td>0</td><td>0.074</td><td>0.959</td><td>0.957</td><td>0.023</td><td>0.311</td></tr><tr><td>Adversarial Training</td><td>Adam-based</td><td>9.65</td><td>90.35</td><td>0.747</td><td>0.421</td><td>2.350</td><td>0.296</td><td>0.149</td></tr><tr><td rowspan="3">with Adam-based Attack Adversarial Training</td><td>PGD</td><td>37.85</td><td>62.15</td><td>0.618</td><td>0.546</td><td>2.350</td><td>0.239</td><td>0.190</td></tr><tr><td>Adam-based</td><td>1.56</td><td>98.44</td><td>0.839</td><td>0.335</td><td>4.626</td><td>0.364</td><td>0.126</td></tr><tr><td>PGD</td><td>4.02</td><td>95.98</td><td>0.781</td><td>0.392</td><td>4.626</td><td>0.331</td><td>0.145</td></tr><tr><td rowspan="3">Adversarial Purification</td><td>Adam-based</td><td>0.39</td><td>91.41</td><td>0.549</td><td>0.183</td><td>0.957</td><td>0.181</td><td>0.048</td></tr><tr><td>PGD</td><td>2.34</td><td>83.98</td><td>0.479</td><td>0.157</td><td>0.957</td><td>0.154</td><td>0.044</td></tr><tr><td>None</td><td></td><td></td><td></td><td></td><td>0.484</td><td>0.568</td><td>0.190</td></tr><tr><td rowspan="4">B</td><td rowspan="3">None</td><td>Adam-based</td><td>99.22</td><td>0.78</td><td>0.244</td><td>0.759</td><td>0.484</td><td>0.241</td><td>0.339</td></tr><tr><td>PGD</td><td>100</td><td>0</td><td>0.157</td><td>0.841</td><td>0.484</td><td>0.201</td><td>0.377</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>0.484</td><td></td><td></td></tr><tr><td>Adversarial Purification</td><td>Adam-based PGD</td><td>0.78 0</td><td>89.1 92.2</td><td>0.587 0.619</td><td>0.273 0.238</td><td>0.484</td><td>0.376 0.411</td><td>0.212 0.201</td></tr></table>

\*When both defense and attack are set to None, Adv.(Syn) refers to the speech synthesized by the TTS system from the source speech.

In adversarial training, the batch-wise random targeted attack strategy is employed, where a batch of speech samples is selected, and each sample within the batch is randomly assigned a target sample with a different speaker identity. Adversarial examples are then generated using the adversarial attack method and combined with the original speeches to fine-tune the speaker encoder model. No data augmentation is applied. The model is optimized using an Adam optimizer with a cosine decay learning rate schedule, starting at 1e-3 and decaying to 1e-5. The batch size is 256, and the number of epochs is 3.

## C. Diffusion-based Adversarial Purification Training

DiffWave [42], a representative diffusion model in the waveform domain, is used as a defensive purification model. We use the same settings as those in [25] for diffusion parameters. The VoxCeleb2 development set is employed for model training, with input utterances truncated to 2 seconds. The learning rate is 2e-4, and the batch size is 16.

We introduce a ResNet18-based binary classifier before the diffusion model to prevent the diffusion model from damaging normal speech. The ArcFace (m=0.2, s=32) classifier is introduced to identification. A subset of the VoxCeleb2 development set is selected, from which 256,000 adversarial samples are generated using adversarial attack methods and split 9:1 for training and testing. The model is updated using the Adam optimizer with a cosine decay learning rate schedule, starting at 1e-3 and decaying to 1e-5. The batch size is 256, and the total number of epochs is 10.

## IV. RESULTS AND ANALYSIS

We used the adversarial attack methods described in II-A to randomly generate 2,560 adversarial samples for each method in the VoxCeleb2 data set and the WenetSpeech4TTS development set, and evaluated the performance of both the traditional and LLM-based zero-shot TTS systems. Table I presents the results for each method in terms of attack efficacy, defense performance, and synthesis quality. We define the defense as successful if the speaker embedding of the adversarial sample is most similar to the source speech’s. Conversely, the attack is successful if the adversarial sample’s speaker embedding is most similar to the target speech’s.

## A. Adversarial Attack Results

Both attack methods exhibited substantial effectiveness, nearly achieving a 100% attack success rate. In model A, after applying the attacks, the cosine similarity between the adversarial samples and the target speech was measured at 0.934 and 0.959, respectively. In contrast, the cosine similarity between the adversarial samples and the source speech decreased to 0.134 and 0.074. Quantitatively, the PGD attack method demonstrated greater effectiveness compared to the Adam-based attack method. The same conclusion can be drawn for Model B.

## B. Adversarial Training Defense Results

After incorporating adversarial samples into the training process, the model’s robustness improved, with defense success rates rising from 0.47% and 0% to 90.35% and 95.98%, respectively. However, the introduction of adversarial samples, as cross-domain data, caused a degree of performance degradation on normal data. The impact of this degradation was more pronounced with stronger attack methods, as evidenced by the EER on the Vox1-O, which increased from 0.957% to 2.35% and 4.626%. Additionally, it is noteworthy that models trained with adversarial samples generated by the weaker Adam-based attack exhibited significantly reduced defense performance against stronger PGD attacks. In contrast, models trained with PGD-generated adversarial samples retained strong defense capabilities against Adam-based attacks, achieving a high defense success rate of 98.44%.

<!-- image-->

(a) Attack and defense success rate across different diffusion steps.  
<!-- image-->  
(b) The similarity of speaker embeddings between the adversarially purified speech and the source and target speech across different diffusion steps.  
Fig. 3. Adversarial purification performance across different diffusion steps.

## C. Adversarial Purification Defense Results

Diffusion-based Adversarial Purification is indeed a promising emerging technology. It can serve as a pre-processing module, providing a robust defense against various attack methods without re-training the speaker encoder and the TTS system. In model A, we observe that after adversarial purification, the attack success rates of adversarial samples decreased from 99.53% and 100% to 0.39% and 2.34%, respectively. However, the defense success rates improved by only 91.41% and 83.98%. This is because the perturbations removed by the diffusion module cannot perfectly match the added perturbations. Consequently, the denoised speaker embedding may resemble those of other speakers rather than the source speaker’s. Moreover, controlling the denoising strength in diffusion-based Adversarial Purification is crucial. As shown in Fig.3, with an increase in diffusion steps, although the success rate of the attack and the similarity to the target speaker decrease, the success rate of the defense and the similarity to the source speaker also eventually decrease after reaching a certain inflection point.

Additionally, it is important to note that the diffusion module can also introduce some damage to normal speech. As shown in Fig.3(b), the similarity between the normal speech and the source speaker rapidly declines with increasing diffusion steps. This degradation can further affect the quality of TTS synthesis. Therefore, we introduced a ResNet18-based discriminator in front of the diffusion model. Experimental results show that, after training on both positive and negative samples, this discriminator achieved a 100% recognition rate for adversarial samples of the attack types it was trained on. We will evaluate unseen adversarial methods in the future.

## D. Zero-Shot TTS Synthesis Results

We also explored the impact of different methods on the quality of zero-shot TTS synthesis. By extracting the speaker embeddings from adversarial samples and synthesizing speech, we evaluated the results by calculating the cosine similarity between the speaker embeddings of the synthesized speech and those of the source and target speakers. We observed that adversarial training defenses resulted in adversarial samples maintaining a high similarity to the source speaker, but with a relatively high similarity to the target speaker as well. In contrast, adversarial purification methods significantly reduced the similarity to the target speaker but also degraded a substantial portion of the source speaker’s information.

## V. CONCLUSION

This paper explores adversarial attacks and robust defenses in speaker embedding based zero-shot TTS systems, including both traditional and LLM-based. In the adversarial attack, we employ PGD and Adam-based white-box attack methods to target the speaker encoder of the zero-shot TTS system, aiming to guide the TTS system into synthesizing speech that closely resembles the target speaker. To mitigate the potential threats these attacks posed, we implemented traditional active defense strategies, such as adversarial training, and novel passive defense strategies based on diffusion models for adversarial purification. We assessed the effectiveness of these defenses, their impact on model performance, and their effects on synthesis quality.

## VI. ACKNOWLEDGEMENT

This research is funded in part by the National Natural Science Foundation of China (62171207), Guangdong Science and Technology Plan (2023A1111120012) and OPPO. Many thanks for the computational resource provided by the Advanced Computing East China Sub-Center.

[1] Yi Ren, Chenxu Hu, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, and Tie-Yan Liu, “Fastspeech 2: Fast and high-quality end-to-end text to speech,” in ICLR. 2021, OpenReview.net.

[2] Jaehyeon Kim, Sungwon Kim, Jungil Kong, and Sungroh Yoon, “Glowtts: A generative flow for text-to-speech via monotonic alignment search,” Advances in Neural Information Processing Systems, vol. 33, pp. 8067–8077, 2020.

[3] Jaehyeon Kim, Jungil Kong, and Juhee Son, “Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech,” in International Conference on Machine Learning. PMLR, 2021, pp. 5530– 5540.

[4] Sercan Arik, Jitong Chen, Kainan Peng, Wei Ping, and Yanqi Zhou, “Neural voice cloning with a few samples,” Advances in neural information processing systems, vol. 31, 2018.

[5] Ye Jia, Yu Zhang, Ron Weiss, Quan Wang, Jonathan Shen, Fei Ren, Patrick Nguyen, Ruoming Pang, Ignacio Lopez Moreno, Yonghui Wu, et al., “Transfer learning from speaker verification to multispeaker text-to-speech synthesis,” Advances in neural information processing systems, vol. 31, 2018.

[6] Yihan Wu, Xu Tan, Bohan Li, Lei He, Sheng Zhao, Ruihua Song, Tao Qin, and Tie-Yan Liu, “Adaspeech 4: Adaptive text to speech in zeroshot scenarios,” in INTERSPEECH. 2022, pp. 2568–2572, ISCA.

[7] Chengyi Wang, Sanyuan Chen, Yu Wu, Ziqiang Zhang, Long Zhou, Shujie Liu, Zhuo Chen, Yanqing Liu, Huaming Wang, Jinyu Li, et al., “Neural codec language models are zero-shot text to speech synthesizers,” arXiv preprint arXiv:2301.02111, 2023.

[8] Sanyuan Chen, Shujie Liu, Long Zhou, Yanqing Liu, Xu Tan, Jinyu Li, Sheng Zhao, Yao Qian, and Furu Wei, “Vall-e 2: Neural codec language models are human parity zero-shot text to speech synthesizers,” arXiv preprint arXiv:2406.05370, 2024.

[9] Weicheng Cai, Jinkun Chen, and Ming Li, “Exploring the encoding layer and loss function in end-to-end speaker and language recognition system,” arXiv preprint arXiv:1804.05160, 2018.

[10] David Snyder, Daniel Garcia-Romero, Gregory Sell, Daniel Povey, and Sanjeev Khudanpur, “X-vectors: Robust DNN embeddings for speaker recognition,” in ICASSP. 2018, pp. 5329–5333, IEEE.

[11] Brecht Desplanques, Jenthe Thienpondt, and Kris Demuynck, “ECAPA-TDNN: emphasized channel attention, propagation and aggregation in TDNN based speaker verification,” in INTERSPEECH. 2020, pp. 3830– 3834, ISCA.

[12] Rosa Gonzalez Hautam ´ aki, Tomi Kinnunen, Ville Hautam ¨ aki, and Anne- ¨ Maria Laukkanen, “Automatic versus human speaker verification: The case of voice mimicry,” Speech Communication, vol. 72, pp. 13–31, 2015.

[13] Jesus Antonio Villalba L ´ opez and Eduardo Lleida, “Detecting replay´ attacks from far-field recordings on speaker verification systems,” in BIOID. 2011, vol. 6583 of Lecture Notes in Computer Science, pp. 274– 285, Springer.

[14] Federico Alegre, Asmaa Amehraye, and Nicholas W. D. Evans, “Spoofing countermeasures to protect automatic speaker verification from voice conversion,” in ICASSP. 2013, pp. 3068–3072, IEEE.

[15] Felix Kreuk, Yossi Adi, Moustapha Cisse, and Joseph Keshet, “Fooling ´ end-to-end speaker verification with adversarial examples,” in ICASSP. 2018, pp. 1962–1966, IEEE.

[16] Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy, “Explaining and harnessing adversarial examples,” in ICLR (Poster), 2015.

[17] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu, “Towards deep learning models resistant to adversarial attacks,” in ICLR (Poster). 2018, OpenReview.net.

[18] Yixiang Wang, Jiqiang Liu, Xiaolin Chang, Jianhua Wang, and Ricardo J Rodr´ıguez, “Ab-fgsm: Adabelief optimizer and fgsm-based approach to generate adversarial examples,” Journal of Information Security and Applications, vol. 68, pp. 103227, 2022.

[19] Nicholas Carlini and David A. Wagner, “Towards evaluating the robustness of neural networks,” in IEEE Symposium on Security and Privacy. 2017, pp. 39–57, IEEE Computer Society.

[20] Yulong Cao, Danfei Xu, Xinshuo Weng, Zhuoqing Mao, Anima Anandkumar, Chaowei Xiao, and Marco Pavone, “Robust trajectory prediction against adversarial attacks,” in Conference on Robot Learning. PMLR, 2023, pp. 128–137.

[21] Haibin Wu, Songxiang Liu, Helen Meng, and Hung-yi Lee, “Defense against adversarial attacks on spoofing countermeasures of ASV,” in ICASSP. 2020, pp. 6564–6568, IEEE.

[22] Florinel-Alin Croitoru, Vlad Hondru, Radu Tudor Ionescu, and Mubarak Shah, “Diffusion models in vision: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 9, pp. 10850– 10869, 2023.

[23] Ju-Ho Kim, Jungwoo Heo, Hyun-seo Shin, Chan-yeong Lim, and Ha-Jin Yu, “Diff-sv: A unified hierarchical framework for noise-robust speaker verification using score-based diffusion probabilistic models,” in ICASSP. 2024, pp. 10341–10345, IEEE.

[24] Yibo Bai, Xiao-Lei Zhang, and Xuelong Li, “Diffusion-based adversarial purification for speaker verification,” IEEE Signal Processing Letters, 2024.

[25] Shutong Wu, Jiongxiao Wang, Wei Ping, Weili Nie, and Chaowei Xiao, “Defending against adversarial audio via diffusion model,” in ICLR. 2023, OpenReview.net.

[26] Diederik P. Kingma and Jimmy Ba, “Adam: A method for stochastic optimization,” in ICLR (Poster), 2015.

[27] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, “Deep residual learning for image recognition,” in CVPR. 2016, pp. 770–778, IEEE Computer Society.

[28] Joon Son Chung, Arsha Nagrani, and Andrew Zisserman, “Voxceleb2: Deep speaker recognition,” in INTERSPEECH. 2018, pp. 1086–1090, ISCA.

[29] Arsha Nagrani, Joon Son Chung, and Andrew Zisserman, “Voxceleb: A large-scale speaker identification dataset,” in INTERSPEECH. 2017, pp. 2616–2620, ISCA.

[30] W. Cai, J. Chen, J. Zhang, and M. Li, “On-the-Fly Data Loader and Utterance-Level Aggregation for Speaker and Language Recognition,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, pp. 1038–1051, 2020.

[31] David Snyder, Guoguo Chen, and Daniel Povey, “Musan: A music, speech, and noise corpus,” arXiv preprint arXiv:1510.08484, 2015.

[32] Tom Ko, Vijayaditya Peddinti, Daniel Povey, Michael L. Seltzer, and Sanjeev Khudanpur, “A study on data augmentation of reverberant speech for robust speech recognition,” in ICASSP. 2017, pp. 5220–5224, IEEE.

[33] Weiqing Wang, Danwei Cai, Xiaoyi Qin, and Ming Li, “The DKU-DukeECE Systems for VoxCeleb Speaker Recognition Challenge 2020,” arXiv.2010.12731.

[34] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou, “Arcface: Additive angular margin loss for deep face recognition,” in CVPR. 2019, pp. 4690–4699, Computer Vision Foundation / IEEE.

[35] Heiga Zen, Viet Dang, Rob Clark, Yu Zhang, Ron J. Weiss, Ye Jia, Zhifeng Chen, and Yonghui Wu, “Libritts: A corpus derived from librispeech for text-to-speech,” in INTERSPEECH. 2019, pp. 1526– 1530, ISCA.

[36] Ilya Loshchilov, Frank Hutter, et al., “Fixing weight decay regularization in adam,” arXiv preprint arXiv:1711.05101, vol. 5, 2017.

[37] Zhihao Du, Jiaming Wang, Qian Chen, Yunfei Chu, Zhifu Gao, Zerui Li, Kai Hu, Xiaohuan Zhou, Jin Xu, Ziyang Ma, et al., “Lauragpt: Listen, attend, understand, and regenerate audio with gpt,” arXiv preprint arXiv:2310.04673, 2023.

[38] Zhihao Du, Shiliang Zhang, Kai Hu, and Siqi Zheng, “Funcodec: A fundamental, reproducible and integrable open-source toolkit for neural speech codec,” in ICASSP. IEEE, 2024, pp. 591–595.

[39] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al., “Qwen technical report,” arXiv preprint arXiv:2309.16609, 2023.

[40] Linhan Ma, Dake Guo, Kun Song, Yuepeng Jiang, Shuai Wang, Liumeng Xue, Weiming Xu, Huan Zhao, Binbin Zhang, and Lei Xie, “Wenetspeech4tts: A 12,800-hour mandarin tts corpus for large speech generation model benchmark,” arXiv preprint arXiv:2406.05763, 2024.

[41] Kangxiang Xia, Dake Guo, Jixun Yao, Liumeng Xue, Hanzhao Li, Shuai Wang, Zhao Guo, Lei Xie, Qingqing Zhang, Lei Luo, et al., “The iscslp 2024 conversational voice clone (covoc) challenge: Tasks, results and findings,” arXiv preprint arXiv:2411.00064, 2024.

[42] Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro, “Diffwave: A versatile diffusion model for audio synthesis,” in ICLR. 2021, OpenReview.net.Done
