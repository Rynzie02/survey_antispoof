
# DEFENDING AGAINST ADVERSARIAL AUDIO VIA DIFFUSION MODEL

Shutong Wu1,2∗ Jiongxiao Wang1 Wei Ping3 Weili Nie3 Chaowei Xiao1 1Arizona State University 2Shanghai Jiao Tong University 3NVIDIA

## ABSTRACT

Deep learning models have been widely used in commercial acoustic systems in recent years. However, adversarial audio examples can cause abnormal behaviors for those acoustic systems, while being hard for humans to perceive. Various methods, such as transformation-based defenses and adversarial training, have been proposed to protect acoustic systems from adversarial attacks, but they are less effective against adaptive attacks. Furthermore, directly applying the methods from the image domain can lead to suboptimal results because of the unique properties of audio data. In this paper, we propose an adversarial purification-based defense pipeline, AudioPure, for acoustic systems via offthe-shelf diffusion models. Taking advantage of the strong generation ability of diffusion models, AudioPure first adds a small amount of noise to the adversarial audio and then runs the reverse sampling step to purify the noisy audio and recover clean audio. AudioPure is a plug-and-play method that can be directly applied to any pretrained classifier without any fine-tuning or re-training. We conduct extensive experiments on speech command recognition task to evaluate the robustness of AudioPure. Our method is effective against diverse adversarial attacks (e.g. L2 or L∞-norm). It outperforms the existing methods under both strong adaptive white-box and black-box attacks bounded by L2 or L∞- norm (up to +20% in robust accuracy). Besides, we also evaluate the certified robustness for perturbations bounded by L2-norm via randomized smoothing. Our pipeline achieves a higher certified accuracy than baselines. Code is available at https://github.com/cychomatica/AudioPure.

## 1 INTRODUCTION

Deep neural networks (DNNs) have demonstrated great successes in different tasks in the audio domain, such as speech command recognition, keyword spotting, speaker identification, and automatic speech recognition. Acoustic systems built by DNNs (Amodei et al., 2016; Shen et al., 2019) are applied in safety-critical applications ranging from making phone calls to controlling household security systems. Although DNN-based models have exhibited significant performance improvement, extensive studies have shown that they are vulnerable to adversarial examples (Szegedy et al., 2014; Carlini & Wagner, 2018; Qin et al., 2019; Du et al., 2020; Abdullah et al., 2021; Chen et al., 2021a), where attackers add imperceptible and carefully crafted perturbations to the original audio to mislead the system with incorrect predictions. Thus, it becomes crucial to design robust DNN-based acoustic systems against adversarial examples.

To address it, existing works (e.g., Rajaratnam & Alshemali, 2018; Yang et al., 2019) have tried to leverage the temporal dependency property of audio to defend against adversarial examples. They apply the time-domain and frequency-domain transformations to the adversarial examples to improve the robustness. Although they can alleviate this problem to some extent, they are still vulnerable against strong adaptive attacks where the attacker obtains full knowledge of the whole acoustic system (Tramer et al., 2020). Another way to enhance the robustness against adversarial examples is adversarial training (Goodfellow et al., 2015; Madry et al., 2018) that adversarial perturbations have been added to the training stage. Although it has been acknowledged as the most effective defense, the training process will require expensive computational resources and the model is still vulnerable to other types of adversarial examples that are not similar to those used in the training process (Tramer & Boneh, 2019).

Adversarial purification (Yoon et al., 2021; Shi et al., 2021; Nie et al., 2022) is another family of defense methods that utilizes generative models to purify the adversarial perturbations of the input examples before they are fed into neural networks. The key of such methods is to design an effective generative model for purification. Recently, diffusion models have been shown to be the state-of-the-art models for images (Song & Ermon, 2019; Ho et al., 2020; Nichol & Dhariwal, 2021; Dhariwal & Nichol, 2021) and audio synthesis (Kong et al., 2021; Chen et al., 2021b). It motivates the community to use it for purification. In particular, in the image domain, DiffPure (Nie et al., 2022) applies diffusion models as purifiers and obtains good performance in terms of both clean and robust accuracy on various image classification tasks. Since such methods do not require training the model with pre-defined adversarial examples, they can generalize to diverse threats. Given the significant progress of diffusion models made in the image domain, it motivates us to ask: is it possible to obtain similar success in the audio domain?

Unlike the image domain, audio signals have some unique properties. There are different choices of audio representations, including raw waveforms and various types of time-frequency representations (e.g., Mel spectrogram, MFCC). When designing an acoustic system, some particular audio representations may be selected as the target features, and defenses that work well on some features may perform poorly on other features. In addition, one may think of treating the 2-D time-frequency representations (i.e., spectrogram) as images, where the frequency-axis is set as height and the timeaxis is set as width, then directly apply the successful DiffPure (Nie et al., 2022) from the image domain for spectrogram. Despite the simplicity, there are two major issues: i) the acoustic system can take audio with variable time duration as the input, while the underlying diffusion model within DiffPure can only handle inputs with fixed width and height. ii) Even if we apply it in a fixed-length segment-wise manner for the time being, it still achieves the suboptimal results as we will demonstrate in this work. These unique issues pose a new challenge of designing and evaluating defense systems in the audio domain.

In this work, we aim to defend against diverse unseen adversarial examples without adversarial training. We propose a play-and-plug purification pipeline named AudioPure based on a pre-trained diffusion model by leveraging the unique properties of audio. In specific, our model consists of two main components: (1) a waveform-based diffusion model and (2) a classifier. It takes the audio waveform as input and leverages the diffusion model to purify the adversarial audio perturbations. Given an adversarial input formatted with waveform, AudioPure first adds a small amount of noise via the diffusion process to override the adversarial perturbations, and then uses the truncated reverse process to recover the clean sample. The recovered sample is fed into the classifier.

We conduct extensive experiments to evaluate the robustness of our method on the task of speech command recognition. We carefully design the adaptive attacks so that the attacker can accurately compute the full gradients to evaluate the effectiveness of our method. In addition, we also comprehensively evaluate the robustness of our method against different black-box attacks and the Expectation Over Transformation (EOT) attack. Our method shows a better performance under both white-box and black-box attacks against diverse adversarial examples. Moreover, we also evaluate the certified robustness of AudioPure via randomized smoothing, which offers a provable guarantee of model robustness against L2-based perturbation. We show that our method achieves better certified robustness than baselines. Specifically, our method obtains a significant improvement (up to +20% at most in robust accuracy) compared to adversarial training, and over 5% higher certified robust accuracy than baselines. To the best of our knowledge, we are the first to use diffusion models to enhance the security of acoustic systems and investigate how different working domains of defenses affect adversarial robustness.

## 2 RELATED WORK

Adversarial attacks and defenses. Szegedy et al. (2014) introduce adversarial examples, which look similar to normal examples but will fool the neural networks to give incorrect predictions. Usually, adversarial examples are constrained by ${ \mathcal { L } } _ { p }$ norm to ensure the imperceptibility. Recently, stronger attack methods are emerging (Madry et al., 2018; Carlini & Wagner, 2017; Andriushchenko et al., 2020; Croce & Hein, 2020; Xiao et al., 2018a;b; 2019; 2022b;a; Cao et al., 2019b;a; 2022a).

<!-- image-->  
Figure 1: The architecture of the whole acoustic system protected by AudioPure (black line in the figure) and the adaptive attack (orange line in the figure). AudioPure first adds noise to the adversarial audio and then runs the reverse process to recover purified audio. Next, the purified audio is transformed into the spectrogram, and the spectrogram is fed into the classifier to get predictions. The attacker updates the adversarial audio based on the gradients backpropagated through SDE. Without AudioPure , the adversarial audio transfers to the spectrogram and feeds into the classifier directly.

In the audio domain, Carlini & Wagner (2018) introduce audio adversarial examples, and Qin et al. (2019) manage to make them more imperceptible. Black-box attacks (Du et al., 2020; Chen et al., 2021a) are also developed, aiming to mislead the end-to-end acoustic systems.

In order to protect neural networks from adversarial attacks, different defense methods are proposed. The most widely used one is adversarial training (Madry et al., 2018), which deliberately uses adversarial examples as the training data of neural networks. The main problems of adversarial training are the accuracy drop of benign examples and the expensive computational cost. Many improved versions of adversarial training aim to alleviate these problems (Wong et al., 2020; Shafahi et al., 2019; Zhang et al., 2019b;a; Sun et al., 2021; Cao et al., 2022b; Zhang et al., 2019c). Another line of work is adversarial purification (Yoon et al., 2021; Shi et al., 2021; Nie et al., 2022), which uses generative models to remove the adversarial perturbations before classification. Both of these two types of defenses are mainly developed for computer vision tasks and cannot be directly applied to the audio domain. In this paper, we explicitly design a defense pipeline according to the characteristics of audio data.

Speech processing. Many speech processing applications are vulnerable to adversarial attacks, including speech command recognition (Warden, 2018), keyword spotting (Chen et al., 2014; Li et al., 2019), speaker identification (Reynolds et al., 2000; Ravanelli & Bengio, 2018; Snyder et al., 2018), and speech recognition (Amodei et al., 2016; Shen et al., 2019; Ravanelli et al., 2019). In particular, speech command recognition is closely related to keyword spotting, and can be viewed as speech recognition with limited vocabulary. In this work, we choose speech command recognition as the testbed for the proposed AudioPure pipeline. The proposed pipeline is applicable for keyword spotting and speech recognition.

A speech command recognition system consists of a feature extractor and a classifier. The feature extractor processes the raw audio waveforms and outputs acoustic features, e.g. Mel spectrograms or Mel-frequency cepstral coefficients (MFCC). Then these features are fed into the classifier, and the classifier gives predictions. Given the 2-D spectrogram features, convolutional neural networks for images are readily applicable (Simonyan & Zisserman, 2015; He et al., 2016; Zagoruyko & Komodakis, 2016; Xie et al., 2017; Huang et al., 2017).

## 3 METHOD

## 3.1 BACKGROUND OF DIFFUSION MODELS

A diffusion model normally consists of a forward diffusion process and a reverse sampling process. The forward diffusion process gradually adds gaussian noise to the input data until the distribution of the noisy data converges to a standard Gaussian distribution. The reverse sampling process takes the standard gaussian noise as input and gradually denoises the noisy data to recover clean data. At present, diffusion models can be divided into two different types: discrete-time diffusion models based on sequential sampling, such as SMLD Song & Ermon (2019), DDPM (Ho et al., 2020), and DDIM (Song et al., 2021a), and continuous-time diffusion models based on SDEs (Song et al., 2021c). Song et al. (2021c) also build the connection between these two types of diffusion models.

Denoising Diffusion Probabilistic Models (DDPM) (Ho et al., 2020) is one of the most widely used diffusion models. Many of the subsequently proposed diffusion models, including DiffWave for audio (Kong et al., 2021), are based on the DDPM formulation. In DDPM, both the diffusion and reverse processes are defined by Markov chains. For input data $\mathbf { x } _ { 0 } \in \mathbb { R } ^ { d }$ , we denote $\mathbf { x } _ { 0 } \sim q ( \mathbf { x } _ { 0 } )$ as the original data distribution, and $\mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { N }$ are intermediate latent variables from the distributions $q ( \mathbf { x } _ { 1 } | \mathbf { x } _ { 0 } ) , \ldots , q ( \mathbf { x } _ { N } | \mathbf { x } _ { N - 1 } )$ , where N is the total number of steps. Generally, with a pre-defined or learned variance schedule $\beta _ { 1 } , \ldots , \beta _ { N }$ (usually linearly increasing small constants), the forward transition probability $q ( \mathbf { x } _ { n } | \mathbf { x } _ { n - 1 } )$ can be formulated as:

$$
q ( \mathbf { x } _ { n } | \mathbf { x } _ { n - 1 } ) = { \mathcal { N } } ( \mathbf { x } _ { n } ; { \sqrt { 1 - \beta _ { n } } } \mathbf { x } _ { n - 1 } , \beta _ { n } \mathbf { I } ) ,\tag{1}
$$

Based on the variance schedule $\{ \beta _ { n } \}$ , a set of constants is defined as:

$$
\alpha _ { n } = 1 - \beta _ { n } , \quad \bar { \alpha } _ { n } = \prod _ { n = 1 } ^ { N } \alpha _ { n } , \quad \tilde { \beta } _ { n } = \left\{ \begin{array} { r r } { \frac { 1 - \bar { \alpha } _ { n - 1 } } { 1 - \bar { \alpha } _ { n } } \beta _ { n } , } & { n > 1 } \\ { \beta _ { 1 } , } & { n = 1 } \end{array} \right. ,\tag{2}
$$

and using the reparameterization trick, we have:

$$
q ( \mathbf { x } _ { n } | \mathbf { x } _ { 0 } ) = \mathcal { N } ( \mathbf { x } _ { n } ; \sqrt { \bar { \alpha } _ { n } } \mathbf { x } _ { 0 } , ( 1 - \bar { \alpha } _ { n } ) \mathbf { I } )\tag{3}
$$

When n gradually gets larger to infinity, $q ( \mathbf { x } _ { n } | \mathbf { x } _ { 0 } )$ will converge to a standard Gaussian distribution. Meanwhile, for the reverse process, we have:

$$
\begin{array} { r } { { \bf x } _ { n - 1 } \sim p _ { \boldsymbol \theta } ( { \bf x } _ { n - 1 } | { \bf x } _ { n } ) = \mathcal { N } ( { \bf x } _ { n - 1 } ; \mu _ { \boldsymbol \theta } ( { \bf x } _ { n } , n ) , \sigma _ { \boldsymbol \theta } ^ { 2 } ( { \bf x } _ { n } , n ) { \bf I } ) , } \end{array}\tag{4}
$$

where the mean term $\mu _ { \boldsymbol { \theta } } ( \mathbf { x } _ { n } , n )$ and the variance term $\sigma _ { \theta } ^ { 2 } ( \mathbf { x } _ { n } , n )$ is instantiated by parameter θ. Ho et al. (2020); Kong et al. (2021) use a neural network $\epsilon _ { \theta }$ to define $\mu _ { \theta }$ , and $\sigma _ { \theta }$ is fixed to a constant:

$$
\mu _ { \theta } ( \mathbf { x } _ { n } , n ) = \frac { 1 } { \sqrt { \alpha _ { n } } } \left( \mathbf { x } _ { n } - \frac { \beta _ { n } } { \sqrt { 1 - \bar { \alpha } _ { n } } } \epsilon _ { \theta } ( \mathbf { x } _ { n } , n ) \right) , \quad \sigma _ { \theta } ( \mathbf { x } _ { n } , n ) = \sqrt { \tilde { \beta } _ { n } } .\tag{5}
$$

We denote $\mathbf { x } _ { n } ( \mathbf { x } _ { 0 } , \epsilon ) = \sqrt { \bar { \alpha } _ { n } } \mathbf { x } _ { 0 } + \sqrt { ( 1 - \bar { \alpha } _ { n } ) } \epsilon , \epsilon \sim \mathcal { N } ( 0 , \mathbf { I } )$ , and the optimization objective is:

$$
\theta ^ { \star } = \arg \operatorname* { m a x } _ { \theta } \sum _ { n = 1 } ^ { N } \lambda _ { n } \mathbb { E } _ { \mathbf { x } ( 0 ) } \left\| \epsilon - \epsilon _ { \theta } \big ( \sqrt { \bar { \alpha } _ { n } } \mathbf { x } _ { 0 } + \sqrt { \big ( 1 - \bar { \alpha } _ { n } \big ) } \epsilon , n \big ) \right\| _ { 2 } ^ { 2 }\tag{6}
$$

where $\lambda _ { n }$ is the weighting coefficient (Ho et al., 2020).

According to Song et al. (2021c), as $N  \infty ,$ , DDPM becomes VP-SDE, a continuous-time formulation of diffusion models. Particularly, the forward SDE is formulated as:

$$
\mathrm { d } \mathbf { x } = - \frac { 1 } { 2 } \beta ( t ) \mathbf { x } \mathrm { d } t + \sqrt { \beta ( t ) } \mathrm { d } \mathbf { w } .\tag{7}
$$

where $t \in [ 0 , 1 ]$ , dt is an infinitesimal positive time step, w is a standard Wiener process, $\beta ( t )$ is the continuous-time noise schedule. Similarly, the reverse SDE can be defined as:

$$
\mathrm { d } \mathbf { x } = - \frac { 1 } { 2 } \beta ( t ) [ \mathbf { x } + 2 \nabla _ { \mathbf { x } } \log p _ { t } ( \mathbf { x } ) ] \mathrm { d } t + \sqrt { \beta ( t ) } \mathrm { d } \bar { \mathbf { w } } ,\tag{8}
$$

where dt is an infinitesimal negative time step, and w¯ is a reverse-time standard Wiener process.

## 3.2 AUDIOPURE: A PLUG-AND-PLAY DEFENSE FOR ACOUSTIC SYSTEMS

To standardize the formulation of the defense, as suggested by Nie et al. (2022), we use the continuous-time formulation defined by Eq. 7 and Eq. 8. Note that since the existing pretrained DiffWave models (Kong et al., 2021) are based on DDPM, we will use their equivalent VP-SDE.

If we use the Euler-Maruyama method to solve the VP-SDE and the step size $\begin{array} { r } { \Delta t { } = \frac { 1 } { N } } \end{array}$ , the sampling of the reverse-time SDE will be equivalent to the reverse sampling of DDPM (detailed proofs can be found in Song et al. (2021c)). Under this prerequisite, we have $\begin{array} { r } { t = \frac { n } { N } } \end{array}$ where $n \in \{ 1 , \bar { , } \ldots , N \}$ . We define $\begin{array} { r } { \beta ( \frac { n } { N } ) : = \beta _ { n } , \bar { \alpha } ( \frac { n } { N } ) : = \bar { \alpha } _ { n } , \tilde { \beta } ( \frac { n } { N } ) : = \tilde { \beta } _ { n } } \end{array}$ , and $\begin{array} { r } { \mathbf { x } ( \frac { n } { N } ) : = \mathbf { x } _ { n } } \end{array}$ Given an adversarial example $x _ { a d v }$ as the input at $t = 0 ,$ , i.e. ${ \bf x } _ { 0 } = { \bf x } _ { a d v } ,$ , we first run the forward SDE from t = 0 to $\begin{array} { r } { t ^ { \star } = \frac { n ^ { \star } } { N } } \end{array}$ by solving Eq. 7 (it is equivalent to running n∗ DPPM steps), which yields:

$$
\mathbf { x } ( t ^ { \star } ) = \sqrt { \bar { \alpha } ( t ^ { \star } ) } \mathbf { x } _ { a d v } + \sqrt { 1 - \bar { \alpha } ( t ^ { \star } ) } \mathbf { z } , \qquad \mathbf { z } \sim \mathcal { N } ( 0 , \mathbf { I } ) ,\tag{9}
$$

Next, we run the truncated reverse SDE from $t = t ^ { \star }$ to $t = 0$ by solving Eq. 8. Similar to Nie et al. (2022), we define an SDE solver sdeint that uses the Euler-Maruyama method, and sequentially takes in six inputs: initial value, drift coefficient, diffusion coefficient, Wiener process, initial time, and end time. The reverse output xˆ(0) at $t = 0$ can be formulated as:

$$
\begin{array} { r } { \hat { \mathbf { x } } ( 0 ) = \mathbf { s d e i n t } ( \mathbf { x } ( t ^ { \star } ) , f _ { r e v } , g _ { r e v } , \bar { \mathbf { w } } , t ^ { \star } , 0 ) . } \end{array}\tag{10}
$$

where the drift and diffusion coefficients are:

$$
f _ { r e v } ( { \bf x } , t ) : = - \frac { 1 } { 2 } \beta ( t ) [ { \bf x } + 2 { \bf s } _ { \theta } ( { \bf x } , t ) ] , \qquad g _ { r e v } ( t ) : = \sqrt { \tilde { \beta } ( t ) } .\tag{11}
$$

Note that we use a diffusion coefficient different from Nie et al. (2022) for the purpose of cleaner output (see the detailed explanation in Section 3.3). Next, we use the discrete-time noise estimator $\epsilon _ { \theta } ( \mathbf { x } _ { n } , n )$ to compute the continuous-time score estimator $s _ { \boldsymbol { \theta } } ( \mathbf { x } , t )$ . By defining $\widetilde { \epsilon } _ { \theta } ( \mathbf { x } ( t ) , t ) : =$ $\begin{array} { r } { \epsilon _ { \theta } ( \mathbf { x } ( \frac { n } { N } ) , n ) = \epsilon _ { \theta } ( \mathbf { x } _ { n } , n ) } \end{array}$ with $\begin{array} { r } { t : = \frac { n } { N } } \end{array}$ , the score function in the reverse VP-SDE can be estimated as:

$$
\mathbf { s } _ { \theta } ( \mathbf { x } , t ) = - \frac { \tilde { \epsilon } _ { \theta } ( \mathbf { x } , t ) } { \sqrt { 1 - \bar { \alpha } ( t ) } } \approx \nabla _ { \mathbf { x } } \log p _ { t } ( \mathbf { x } ) .\tag{12}
$$

Accordingly, $\hat { \mathbf { x } } ( 0 )$ , the purified output of the adversarial input ${ \bf x } ( 0 ) = { \bf x } _ { a d v }$ , is fed into the later stages of the acoustic system to make predictions. The whole purification operation can be denoted as a function Purifier : $\mathbb { R } ^ { d } \times \mathbb { R } \to \bar { \mathbb { R } ^ { d } }$ :

$$
\mathbf { P u r i f i e r } ( \mathbf { x } _ { a d v } , n ^ { \star } ) = \mathbf { s d e i n t } \left( { \sqrt { \bar { \alpha } ( { \frac { n ^ { \star } } { N } } ) } } \mathbf { x } _ { a d v } + { \sqrt { 1 - { \bar { \alpha } } ( { \frac { n ^ { \star } } { N } } ) } } \mathbf { z } , f _ { r e v } , g _ { r e v } , { \bar { \mathbf { w } } } , { \frac { n ^ { \star } } { N } } , 0 \right)\tag{13}
$$

The acoustic systems are usually built on the features extracted from the raw audio. For example, the system can extract Mel spectrogram as the features: 1) it first applies short-time Fourier transformation (STFT) on the time-domain waveform to get linear-scale spectrogram, and 2) it then rescales the frequency band to the Mel-scale. We denote this process as Wave2Mel : $\mathbb { R } ^ { d }  \mathbb { R } ^ { m } \times \mathbb { R } ^ { n }$ ? which is a differentiable function. Then the classifier $\bar { F ^ { \prime } } : \mathbb { R } ^ { m } \times \mathbb { R } ^ { n } \to \mathbb { R } ^ { c }$ (usually a convolutional network) takes the Mel spectrogram as the input and gives predictions.

Since both the time domain waveform and time-frequency domain spectrogram go through the pipeline, the purifier can be applied in either the time domain or time-frequency domain. If the purifier is applied in the time domain, the whole defended acoustic system $\bar { \bf A } \bar { \bf S } : \bar { \mathbb { R } ^ { d } } \times \mathbb { R }  \mathbb { R } ^ { c }$ can be formulated as:

$$
\mathbf { A S } ( \mathbf { x } _ { a d v } , n ^ { \star } ) = F ( \mathbb { W } \mathrm { a v e } 2 \mathbf { M e l } ( \mathbf { P u r i f i e r } ( \mathbf { x } _ { a d v } , n ^ { \star } ) ) )\tag{14}
$$

where the waveform Purifier is based on DiffWave.

Meanwhile, if we want to purify the input adversarial examples in the time-frequency domain, we can choose a diffusion model used for image synthesis, and apply it to the output spectrogram of Wave2Mel. We denote this purifier as Purifier $\dot { \mathbf { \sigma } } _ { s p e c } : \mathbb { R } ^ { m } \times \mathbb { R } ^ { \hat { n } } \times \mathbf { \bar { \mathbb { R } } }  \mathbb { R } ^ { m } \times \bar { \mathbb { R } } ^ { n }$ . In this scenario, the whole defended acoustic system will be:

$$
\mathbf { A S } ( \mathbf { x } _ { a d v } , n ^ { \star } ) = F ( \mathbf { P u r i f i e r } _ { s p e c } ( \mathbb { W } \mathrm { a v e } 2 \mathbf { M e l } ( \mathbf { x } _ { a d v } ) ) , n ^ { \star } )\tag{15}
$$

The architecture of the whole pipeline is illustrated in Figure. 1. For the purification in the timefrequency domain spectrogram, we use an Improved DDPM (Nichol & Dhariwal, 2021) trained on the Mel spectrograms of audio data and denote it as $D i f f S p e c$ . We compare these two purifiers and discover that the purification in the time domain waveform is more effective to defend against adversarial audio. Detailed experimental results can be found in Sec. 4.2.

## 3.3 TOWARDS EVALUATING AUDIOPURE

Adaptive attack For the forward diffusion process formulated as Eq. 9, the gradients of the output $\mathbf { x } ( t ^ { \star } )$ w.r.t. the input x(0) is a constant. For the reverse process formulated as Eq. 10, the adjoint method (Li et al., 2020) is applied to compute the full gradients of the objective function L w.r.t. $x ( t ^ { \star } )$ without any out-of-memory issues, by solving another augmented SDE:

$$
\begin{array} { r } { \left( \begin{array} { c } { \mathbf { x } \big ( t ^ { \star } \big ) } \\ { \partial \Sigma } \end{array} \right) = \mathbf { s d e i n t } \left( \left( \begin{array} { c } { \mathbf { x } \big ( 0 \big ) } \\ { \frac { \partial \mathcal { L } } { \partial \mathbf { x } ( 0 ) } } \end{array} \right) , \left( \begin{array} { c } { f _ { r e v } } \\ { \frac { \partial f _ { r e v } } { \partial \mathbf { x } } \mathbf { z } } \end{array} \right) , \left( \begin{array} { c } { g _ { r e v } \mathbf { 1 } } \\ { \mathbf { 0 } } \end{array} \right) , \left( \begin{array} { c } { - \mathbf { w } \big ( 1 - t \big ) } \\ { - \mathbf { w } \big ( 1 - t \big ) } \end{array} \right) , 0 , t ^ { \star } \right) } \end{array}\tag{16}
$$

where 1 and 0 represent the vectors of all ones and all zeros, respectively.

SDE modifications for clean output We observe that directly applying the framework of Nie et al. (2022) to the audio domain will cause the performance degradation. That is, when converting the discrete-time reverse process of DiffWave (Kong et al., 2021) to its corresponding reverse VP-SDE in Eq. 8, the output audio still contains much noise, resulting in lower classification accuracy. We identify two influencing factors and solve this problem by modifying the SDE formulation.

The first factor is the diffusion error due to the mismatch of the reverse variance between the discrete and continuous cases. Ho et al. (2020) observed that both $\sigma _ { \theta } ^ { 2 } = \tilde { \beta } _ { t }$ and $\sigma _ { \theta } ^ { 2 } = \beta _ { t }$ get similar results experimentally in the image domain. However, we find that it is not the case in the audio modeling with diffusion models. For audio synthesis using DiffWave trained with $\sigma _ { \theta } ^ { 2 } = \tilde { \beta } _ { t }$ , if we switch the reverse variance schedule to $\sigma _ { \theta } ^ { 2 } = \dot { \beta } _ { t }$ , the output audio becomes noisy. Thus, in Sec. 3.2 we define $\begin{array} { r } { \tilde { \beta } ( \frac { n } { N } ) = \tilde { \beta } _ { n } } \end{array}$ and use the diffusion coefficient $g _ { r e v } = \sqrt { { \tilde { \beta } } ( t ) }$ in Eq. 11 instead of $g _ { r e v } = \sqrt { \beta ( t ) }$ to match the variance $\tilde { \beta } _ { t }$ in DiffWave.

The second factor is the inaccuracy from the continuous-time noise schedule $\beta ( t ) = \beta _ { 0 } + ( \beta _ { N } - \beta _ { 0 } ) t$ and $\tilde { \alpha } ( t ) = e ^ { - \int _ { 0 } ^ { t } \beta ( s ) \mathrm { d } s }$ used by Nie et al. (2022). The impact of the difference between $\beta ( t ) =$ $\beta _ { 0 } + ( { \dot { \beta } } _ { N } - \beta _ { 0 } ) t$ and $\beta _ { N t }$ t cannot be negligible, especially when N is not large enough (e.g. $N \stackrel { \cdot } { = } 2 0 0$ for the pretrained DiffWave model we use). Besides, when t is close to $0 , \tilde { \alpha } ( t ) = e ^ { - \int _ { 0 } ^ { t } \beta ( s ) \mathrm { d } s }$ is not a good approximation of $\hat { \alpha } _ { N t }$ any more. Thus, we define the continuous-time noise schedule directly based on the discrete schedule, namely, $\textstyle \beta ( { \frac { n } { N } } ) : = \beta _ { n }$ and $\textstyle { \bar { \alpha } } { \big ( } { \frac { n } { N } } { \big ) } : = { \bar { \alpha } } _ { n }$ , for the purpose of better denoised output and more accurate gradient computation.

## 4 EXPERIMENTS

In this section, we first introduce the detailed experimental settings. Then we compare the performance of our method and other defenses under strong white-box adaptive attack where the attacker has full knowledge about the defense and black-box attacks. To further show the robustness of our method, we also evaluate the certified accuracy via randomize smoothing Cohen et al. (2019), which provides a provable guarantee of model robustness against $\mathcal { L } _ { 2 }$ norm bounded adversarial perturbations.

## 4.1 EXPERIMENTAL SETTINGS

Dataset. Our method is evaluated on the task of speech command recognition. We use the Speech Commands dataset (Warden, 2018), which consists of 85,511 training utterances, 10,102 validation utterances, and 4,890 tests utterances. Following the setting of Kong et al. (2021), we choose the utterances which stand for digits $0 \sim 9$ and denote this subset as SC09.

Models. We use DiffWave (Kong et al., 2021) and DiffSpec (based on Improved DDPM (Nichol & Dhariwal, 2021)) as our defensive purifiers, which are representative diffusion models on the waveform domain and the spectral domain respectively. We use the unconditional version of Diffwave with the officially provided pretrained checkpoints. Since the Improved DDPM model does not provide the pretrained checkpoint for audio, we train it from scratch on the Mel spectrograms of audio from SC09. The training details and hyperparameters are in the appendix A. For the classifier, we use ResNeXt-29-8-64(Xie et al., 2017) for spectrogram representation and M5 Net(Dai et al., 2017) for waveform except the experiments for ablation studies.

Table 1: Performance against adaptive attacks among different methods.
<table><tr><td rowspan="2">Defense</td><td rowspan="2">Clean</td><td colspan="6"> $\scriptstyle { \mathcal { L } } _ { \infty }$ </td><td colspan="4"> $\mathcal { L } _ { 2 }$  white-box</td><td rowspan="2">L∞ black-box FAKEBOB</td></tr><tr><td> $\mathrm { P G D } _ { 1 0 }$ </td><td> $\mathrm { P G D _ { 2 0 } }$ </td><td> $\mathrm { P G D } _ { 3 0 }$ </td><td>PD50</td><td> $\mathrm { P G D } _ { 7 0 }$ </td><td> $\mathrm { P G D _ { 1 0 0 } }$ </td><td> $\mathrm { P G D } _ { 1 0 }$  PGD20</td><td> $\mathrm { P G D } _ { 3 0 }$ </td><td> $\mathrm { P G D } _ { 5 0 }$ </td><td> $\mathrm { P G D _ { 1 0 0 } }$ </td></tr><tr><td>None</td><td>100</td><td>3</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0</td><td>0</td><td>0</td><td>21</td></tr><tr><td>AS (Yang et al., 2019)</td><td>100</td><td>4</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0</td><td>0</td><td>0</td><td>24</td></tr><tr><td>MS (Yang et al., 2019)</td><td>100</td><td>6</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0</td><td>0</td><td>0</td><td>21</td></tr><tr><td>DS (Yang et al., 2019)</td><td>99</td><td>2</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0</td><td>0</td><td>0</td><td>16</td></tr><tr><td>LPF (Rajaratnam &amp; Alshemali, 2018)</td><td>100</td><td>5</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0</td><td>0</td><td>0</td><td>20</td></tr><tr><td>BPF (Rajaratnam &amp; Alshemali, 2018)</td><td>99</td><td>5</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0</td><td>0</td><td>0</td><td>18</td></tr><tr><td>AdvTr (Madry et al., 2018)</td><td>100</td><td>86</td><td>79</td><td>74 85</td><td></td><td>71</td><td>73</td><td>70</td><td>68</td><td>65</td><td>65</td><td>92</td></tr><tr><td>AudioPure</td><td>97</td><td>89</td><td>89</td><td></td><td></td><td>84</td><td>89</td><td>86</td><td>83</td><td>85</td><td>84</td><td>86</td></tr></table>

Attacks. For white-box attacks, we use PGD (Madry et al., 2018) with different iteration steps from 10 to 100 among $L _ { \infty }$ and $L _ { 2 }$ norms. The attack budget is set to $\epsilon = 0 . 0 0 2$ for $\mathcal { L } _ { \infty }$ -norm constraint except the ablation study and $\epsilon = 0 . 2 5 3$ for $L _ { 2 }$ norm constraint. For black-box attacks, we apply a query-based attack, FAKEBOB (Chen et al., 2021a), and set the iteration steps to 200, NES samples to 200, and the confidence coefficient $\kappa = 0 . 5$

Baselines. We compare our method with two types of baselines including: (1) transformation-based defense (Yang et al., 2019; Rajaratnam & Alshemali, 2018) including average smoothing (AS), median smoothing (MS), downsampling (DS), low-pass filter (LPF), and band-pass filter (BPF), and (2) adversarial training based defense (AdvTr) (Madry et al., 2018). For adversarial training, we follow the setting of Chen et al. (2022), using $\mathcal { L } _ { \infty } \mathrm { P G D _ { 1 0 } }$ with $\epsilon = 0 . 0 0 2$ and $r a t i o = 0 . 5$

## 4.2 MAIN RESULTS

We evaluate AudioPure $( n ^ { \star } { = } 3$ by default) under adaptive attacks, assuming the attacker obtains full knowledge of our defense. We use the adaptive attack algorithm described in the previous section so that the attacker is able to accurately compute the full gradients for attacking. The results are shown in Table 1. We find that the baseline transformation-based defenses (Yang et al., 2019; Rajaratnam & Alshemali, 2018), including average smoothing (AS), median smoothing (MS), downsampling (DS), low-pass filter (LPF), and band-pass filter (BPF), are virtually broken through by white-box attacks with up to 4% robust accuracy. For the adversarial training-based method (AdvTr) trained on $\mathcal { L } _ { \infty }$ -norm adversarial examples, although it achieves 71% robust accuracy against $\mathcal { L } _ { \infty }$ -based adversarial examples, such the method does not work so well on other types of adversarial examples (i.e., $\mathcal { L } _ { 2 }$ -based method), achieving 65% robust accuracy under $\mathcal { L } _ { 2 } .$ -based $\mathrm { P D G } _ { 1 0 0 }$ attack. Compared with all baselines, AudioPure can obtain much higher robust accuracy, about 10% improvements on average, on $\mathcal { L } _ { \infty }$ -based adversarial examples, and is equally effective against $\mathcal { L } _ { 2 }$ -based white-box attacks, achieving 84% robust accuracy.

We also evaluate AudioPure on black-box attacks including: FAKEBOB (Chen et al., 2021a) and transferability-based attacks. The results of FAKEBOB are shown in Table 1, indicating that our method can keep effective under the query-based black-box attack. The results of the transferabilitybased attack are in the appendix B. They draw the same conclusion. These results further verify the effectiveness of our method. All results indicate that AudioPure can work under diverse attacks with different types of constraints, while adversarial training has to apply different training strategies and re-train the model, making it less effective among unseen attacks than our method. We report the actual inference time in Appendix J and compare out method with more existing methods in Appendix F, G and H. Additionally, we conduct experiments on the Qualcomm Keyword Speech Dataset (Kim et al., 2019), and the results and details are in Appendix E. In this dataset, our method is still effective against adversarial examples.

## 4.3 ABLATION STUDY

PGD steps To ensure the effectiveness of PGD attacks, we test different iteration steps from 10 to 150. As Figure. 2a illustrates, the robust accuracy converges after iteration steps $n \geq 7 0$

Effectiveness against Expectation over transformation (EOT) attack. Besides, since the diffusion and reverse process of AudioPure consist of many randomized sampling operations, we apply the expectation over transformation (EOT) attack to evaluate the effectiveness of AudioPure with different EOT sample sizes. Figure 2b demonstrates the result. We find that AudioPure is effective among different EOT sizes.

<!-- image-->  
(a) robust accuracy with different PGD steps

<!-- image-->  
(b) robust accuracy with different EOT size  
Figure 2: The performance of baseline (no defense, denoted as None), adversarial training (denoted as AdvTr), and AudioPure under attacks with different iteration steps and EOT size. (a) indicates the step of convergence, and the attack is almost optimal when iterating over 70 steps. (b) shows that increasing EOT size can barely affect the robustness of our method.

Table 2: The robust accuracy under $\mathrm { P G D _ { 1 0 } }$ with different attack budget ϵ when using different reverse steps $n ^ { \star }$ . Larger ϵ requires larger $n ^ { \star }$ to ensure better robustness.
<table><tr><td rowspan="2">Attack Budget</td><td colspan="7">Diffusion Steps</td></tr><tr><td> $n ^ { \star } = 0$ </td><td> $n ^ { \star } = 1$ </td><td> $n ^ { \star } = 2$ </td><td> $n ^ { \star } = 3$ </td><td> $\dot { \mathbf { \rho } } _ { n } \star \mathbf { \rho } = 5$ </td><td> $n ^ { \star } = 7$ </td><td> $n ^ { \star } = 1 0$ </td></tr><tr><td> $\epsilon = 0 . 0 0 2$ </td><td>30</td><td>94</td><td>90</td><td>89</td><td>84</td><td>77</td><td>67</td></tr><tr><td> $\epsilon = 0 . 0 0 4$ </td><td></td><td>76</td><td>89</td><td>86</td><td>83</td><td>74</td><td>66</td></tr><tr><td> $\epsilon = 0 . 0 0 8$ </td><td>0</td><td>27</td><td>70</td><td>85</td><td>84</td><td>74</td><td>68</td></tr><tr><td> $\epsilon = 0 . 0 1 6$ </td><td>0</td><td>0</td><td>21</td><td>53</td><td>69</td><td>57</td><td>63</td></tr></table>

Table 3: Ablation studies among different model architectures. The robust accuracy is evaluated under $\mathcal { L } _ { \infty } .$ $\mathrm { P G D } _ { 7 0 }$ . Our method is effective on various models with different architectures.
<table><tr><td rowspan="2">Defense</td><td colspan="2">ResNeXt-29-8-64</td><td rowspan="2">VGG-19-BN Clean</td><td rowspan="2">Robust</td><td rowspan="2">WideResNet-28-10 Clean</td><td rowspan="2">Robust</td><td rowspan="2">DenseNet-BC-100-12 Clean Robust</td><td rowspan="2">Clean</td><td rowspan="2">M5 Robust</td></tr><tr><td>Clean</td><td>Robust</td></tr><tr><td>None</td><td>100</td><td>1</td><td>100</td><td>2</td><td>100</td><td>1</td><td>100</td><td>5</td><td>94</td><td>12</td></tr><tr><td>AudioPure</td><td>97</td><td>84</td><td>99</td><td>81</td><td>99</td><td>85</td><td>96</td><td>79</td><td>94</td><td>70</td></tr></table>

Attack budget ϵ. We evaluate the effectiveness of our method among different ϵ including ϵ = {0.002, 0.004, 0.008, 0.016}. Since the diffusion steps $n ^ { \star }$ are the hyperparameters for AudioPure , we conduct experiments among different $n ^ { \star }$ . As shown in Table 2, if $n ^ { \star }$ is larger than 2, AudioPure will show strong effectiveness among different ϵ. When ϵ increases, it requires a larger $n ^ { \star }$ to achieve the optimal robustness since a larger adversarial perturbation requires a large noise from the forward process of the diffusion model to override the adversarial perturbations and the corresponding larger step to recover purified audio. However, if the $n ^ { \star }$ is too large, it will override the original audio information as well so that the recovered audio from the diffusion model will lose the original audio information, contributing to the performance drop. Furthermore, we explore the extent of the diffusion model for purification in Appendix I.

Architectures. Moreover, we apply AudioPure to different classifiers, including spectrogrambased classifier: VGG-19-BN(Simonyan & Zisserman, 2015), ResNeXt-29-8-64(Xie et al., 2017), WideResNet-28-10(Zagoruyko & Komodakis, 2016) DenseNet-BC-100-12(Huang et al., 2017) and wave-form based classifier: M5 (Dai et al., 2017). Table 3 shows that our method is effective for various neural network classifiers.

Audio representations Audio has different types of representations including raw waveforms or time-frequency representations (e.g., Mel spectrogram). We conduct an ablation study to show the effectiveness of diffusion models by using different representations, including DiffWave, a diffusion model for waveforms (Kong et al., 2021) and DiffSpec, a diffusion model for spectrogram based on the original image model (Nichol & Dhariwal, 2021). The results are shown in Table 4. We find that the DiffWave consistently outperforms DiffSpec against $\mathcal { L } _ { 2 }$ and $\mathcal { L } _ { \infty }$ -based adversarial examples. Moreover, compared with DiffWave, despite DiffSpec achieve higher clean accuracy, it only achieves 49% robust accuracy, a significant 35% performance drop against $\mathcal { L } _ { \mathrm { 2 } }$ -based adversarial examples. We think the potential reason is that the short-time Fourier transform (STFT) is an operation of information compression. The spectrogram contains much less information than the raw audio waveform. This experiment shows that the domain difference contributes to significantly different results, and directly applying the method from the image domain can lead to suboptimal performance for audio. It also verifies the crucial design of AudioPure for adversarial robustness.

Table 4: Ablation studies among different audio representations. We implement AudioPure using two different diffusion models as purifiers, DiffWave and DiffSpec, that respectively process the representations in the time domain and time-frequency domain.
<table><tr><td rowspan="2">Defense</td><td rowspan="2">Clean</td><td colspan="6"> $\mathcal { L } _ { \infty }$  white-box</td><td colspan="5"> $\mathscr { L } _ { 2 } \mathrm { ~ w h i t e { - } b o x }$ </td></tr><tr><td> $\mathrm { P G D } _ { 1 0 }$ </td><td> $\mathrm { P G D _ { 2 0 } }$ </td><td> $\mathrm { P G D _ { 3 0 } }$ </td><td> $\mathrm { P G D } _ { 5 0 }$ </td><td> $\mathrm { P G D } _ { 7 0 }$ </td><td> $\mathrm { P G D } _ { 1 0 0 }$ </td><td> $\mathrm { P G D } _ { 1 0 }$ </td><td> $\mathrm { P G D _ { 2 0 } }$ </td><td> $\mathrm { P G D _ { 3 0 } }$ </td><td> $\mathrm { P G D } _ { 5 0 }$ </td><td> $\mathrm { P G D } _ { 1 0 0 }$ </td></tr><tr><td>DiffWave</td><td>97</td><td>89</td><td>89</td><td>89</td><td>85</td><td>84</td><td>84</td><td>89</td><td>86</td><td>83</td><td>85</td><td>84</td></tr><tr><td>DiffSpec</td><td>99</td><td>92</td><td>84</td><td>78</td><td>75</td><td>72</td><td>71</td><td>74</td><td>62</td><td>58</td><td>54</td><td>49</td></tr></table>

Table 5: Certified accuracy for different methods. For each noise level σ, we add the same level of noise to train the classifier and apply it to RS-Gaussian.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Noise level</td><td colspan="7">Certified radius  $( \mathcal { L } _ { 2 } )$ </td></tr><tr><td>0</td><td>0.25</td><td>0.50</td><td>0.75 1.02</td><td></td><td>1.25</td><td>1.50</td></tr><tr><td rowspan="2">RS-Vanilla</td><td> $\sigma = 0 . 5$ </td><td>30</td><td>21</td><td>12</td><td>6</td><td>4</td><td>3</td><td>3</td></tr><tr><td> $\sigma = 1 . 0$ </td><td>8</td><td>8</td><td>8</td><td>7</td><td>4</td><td>3</td><td>3</td></tr><tr><td rowspan="2">RS-Gaussian</td><td> $\sigma = 0 . 5$ </td><td>49</td><td>39</td><td>33</td><td>23</td><td>14</td><td>6</td><td>3</td></tr><tr><td> $\sigma = 1 . 0$ </td><td>18</td><td>15</td><td>11</td><td>10</td><td>5</td><td>5</td><td>4</td></tr><tr><td rowspan="2">AudioPure</td><td> $\sigma = 0 . 5$ </td><td>45</td><td>40</td><td>35</td><td>27</td><td>21</td><td>17</td><td>13</td></tr><tr><td> $\sigma = 1 . 0$ </td><td>27</td><td>22</td><td>16</td><td>15</td><td>12</td><td>11</td><td>8</td></tr></table>

## 4.4 CERTIFIED ROBUSTNESS

In this section, we evaluate the certified robustness of AudioPure via randomized smoothing(Cohen et al., 2019). Here we draw N = 100, 000 noise samples and select noise levels $\sigma \in \{ 0 . 5 , 1 . 0 \}$ for certification. Note that we follow the same setting from Carlini et al. (2022) and choose to use the one-shot denoising method. The detailed implementation of our method could be found in Appendix C. We compare our results with randomized smoothing using the vanilla classifier and Gaussian augmented classifier, denoted RS-Vanilla and RS-Gaussian respectively. The results are shown in Table 5. We also provide the certified robustness under different $\mathcal { L } _ { 2 }$ perturbation budget with different Gaussian noise $\sigma = \{ 0 . 5 , 1 . 0 \}$ in Figure A of Appendix C. By observing our results, we find that our method outperforms baselines for a better certified accuracy except $\sigma = 0 . 5$ at 0 radius. We also notice that the performance of our method will be even better when the input noise gets larger. This may be due to AudioPure can still recover the clean audio with a large $\mathcal { L } _ { 2 } .$ -based perturbation while Gaussian augmented model could even not be converged when training with such large noise.

## 5 CONCLUSION

In this paper, we propose an adversarial purification-based defense pipeline for acoustic systems. To evaluate the effectiveness of AudioPure , we design the adaptive attack method and evaluate our method among adaptive attacks, EOT attacks, and black-box attacks. Comprehensive experiments indicate that our defense is more effective than existing methods (including adversarial training) among the diverse type of adversarial examples. We show AudioPure achieves better certifiable robustness via Randomized Smoothing than other baselines. Moreover, our defense can be a universal plug-and-play method for classifiers with different architectures.

Limitations. AudioPure introduces the diffusion model, which increases the time and computational cost. Thus, how to improve time and computational efficiency is an important future work. For example, it is interesting to investigate the distillation technique (Salimans & Ho, 2022) and fast sampling method (Kong & Ping, 2021) to reduce the computation complexity introduced by diffusion models.

## ACKNOWLEDGMENT

We thank Prof. Xiaolin Huang from Shanghai Jiao Tong University for the valuable discussions. Shutong Wu is partially supported by the National Natural Science Foundation of China (61977046), Shanghai Science and Technology Program (22511105600), and Shanghai Municipal Science and Technology Major Project (2021SHZDZX0102).

## ETHICS STATEMENT

Our work proposes a defense pipeline for protecting acoustic systems from adversarial audio examples. In particular, our study focuses on speech command recognition, which is closely related to keyword spotting systems. Such systems are well known to be vulnerable to adversarial attacks. Our pipeline will enhance the security aspect of such real-world acoustic systems and benefit the social beings. The Speech Commands dataset used in our study are released by others and has been publicly available for years. The dataset contains various voices from anonymous speakers. To the best of our knowledge, it does not contain any privacy-related information for these speakers.

## REFERENCES

Hadi Abdullah, Muhammad Sajidur Rahman, Washington Garcia, Kevin Warren, Anurag Swarnim Yadav, Tom Shrimpton, and Patrick Traynor. Hear” no evil”, see” kenansville”\*: Efficient and transferable black-box attacks on speech recognition and voice identification systems. In 2021 IEEE Symposium on Security and Privacy (SP), pp. 712–729. IEEE, 2021.

Dario Amodei, Sundaram Ananthanarayanan, Rishita Anubhai, Jingliang Bai, Eric Battenberg, Carl Case, Jared Casper, Bryan Catanzaro, Qiang Cheng, Guoliang Chen, et al. Deep speech 2: End-toend speech recognition in english and mandarin. In International conference on machine learning, pp. 173–182. PMLR, 2016.

Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion, and Matthias Hein. Square attack: a query-efficient black-box adversarial attack via random search. In European Conference on Computer Vision, pp. 484–501. Springer, 2020.

Yulong Cao, Chaowei Xiao, Benjamin Cyr, Yimeng Zhou, Won Park, Sara Rampazzi, Qi Alfred Chen, Kevin Fu, and Z Morley Mao. Adversarial sensor attack on lidar-based perception in autonomous driving. CCS, 2019a.

Yulong Cao, Chaowei Xiao, Dawei Yang, Jing Fang, Ruigang Yang, Mingyan Liu, and Bo Li. Adversarial objects against lidar-based autonomous driving systems. arXiv preprint arXiv:1907.05418, 2019b.

Yulong Cao, Chaowei Xiao, Anima Anandkumar, Danfei Xu, and Marco Pavone. Advdo: Realistic adversarial attacks for trajectory prediction. In European Conference on Computer Vision, pp. 36–52. Springer, 2022a.

Yulong Cao, Danfei Xu, Xinshuo Weng, Zhuoqing Mao, Anima Anandkumar, Chaowei Xiao, and Marco Pavone. Robust trajectory prediction against adversarial attacks. CORL (Oral), 2022b.

Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In 2017 ieee symposium on security and privacy (sp), pp. 39–57. Ieee, 2017.

Nicholas Carlini and David Wagner. Audio adversarial examples: Targeted attacks on speech-totext. In 2018 IEEE security and privacy workshops (SPW), pp. 1–7. IEEE, 2018.

Nicholas Carlini, Florian Tramer, J Zico Kolter, et al. (certified!!) adversarial robustness for free! arXiv preprint arXiv:2206.10550, 2022.

Guangke Chen, Sen Chenb, Lingling Fan, Xiaoning Du, Zhe Zhao, Fu Song, and Yang Liu. Who is real bob? adversarial attacks on speaker recognition systems. In 2021 IEEE Symposium on Security and Privacy (SP), pp. 694–711. IEEE, 2021a.

Guangke Chen, Zhe Zhao, Fu Song, Sen Chen, Lingling Fan, Feng Wang, and Jiashui Wang. Towards understanding and mitigating audio adversarial examples for speaker recognition. IEEE Transactions on Dependable and Secure Computing, 2022.

Guoguo Chen, Carolina Parada, and Georg Heigold. Small-footprint keyword spotting using deep neural networks. In 2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 4087–4091. IEEE, 2014.

Nanxin Chen, Yu Zhang, Heiga Zen, Ron J Weiss, Mohammad Norouzi, and William Chan. Wavegrad: Estimating gradients for waveform generation. In International Conference on Learning Representations, 2021b.

Jeremy Cohen, Elan Rosenfeld, and Zico Kolter. Certified adversarial robustness via randomized smoothing. In International Conference on Machine Learning, pp. 1310–1320. PMLR, 2019.

Francesco Croce and Matthias Hein. Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. In International conference on machine learning, pp. 2206– 2216. PMLR, 2020.

Wei Dai, Chia Dai, Shuhui Qu, Juncheng Li, and Samarjit Das. Very deep convolutional neural networks for raw waveforms. In 2017 IEEE international conference on acoustics, speech and signal processing (ICASSP), pp. 421–425. IEEE, 2017.

Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in Neural Information Processing Systems, 34:8780–8794, 2021.

Chris Donahue, Julian McAuley, and Miller Puckette. Adversarial audio synthesis. In International Conference on Learning Representations, 2018.

Tianyu Du, Shouling Ji, Jinfeng Li, Qinchen Gu, Ting Wang, and Raheem Beyah. Sirenattack: Generating adversarial audio for end-to-end acoustic systems. In Proceedings of the 15th ACM Asia Conference on Computer and Communications Security, pp. 357–369, 2020.

Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. In International Conference on Learning Representations, 2015.

Shixiang Gu and Luca Rigazio. Towards deep neural network architectures robust to adversarial examples. arXiv preprint arXiv:1412.5068, 2014.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020.

Judy Hoffman, Daniel A Roberts, and Sho Yaida. Robust learning with jacobian regularization. arXiv preprint arXiv:1908.02729, 2019.

Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4700–4708, 2017.

Sonal Joshi, Saurabh Kataria, Yiwen Shao, Piotr Zelasko, Jesus Villalba, Sanjeev Khudanpur, and Najim Dehak. Defense against adversarial attacks on hybrid speech recognition using joint adversarial fine-tuning with denoiser. arXiv preprint arXiv:2204.03851, 2022.

Byeonggeun Kim, Mingu Lee, Jinkyu Lee, Yeonseok Kim, and Kyuwoong Hwang. Query-byexample on-device keyword spotting, 2019.

Zhifeng Kong and Wei Ping. On fast sampling of diffusion probabilistic models. arXiv preprint arXiv:2106.00132, 2021.

Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile diffusion model for audio synthesis. In International Conference on Learning Representations, 2021.

Juncheng Li, Shuhui Qu, Xinjian Li, Joseph Szurley, J Zico Kolter, and Florian Metze. Adversarial music: Real world audio adversary against wake-word detection system. Advances in Neural Information Processing Systems, 32, 2019.

Xuechen Li, Ting-Kam Leonard Wong, Ricky TQ Chen, and David Duvenaud. Scalable gradients for stochastic differential equations. In International Conference on Artificial Intelligence and Statistics, pp. 3870–3882. PMLR, 2020.

Yi Luo and Nima Mesgarani. Conv-tasnet: Surpassing ideal time–frequency magnitude masking for speech separation. IEEE/ACM transactions on audio, speech, and language processing, 27(8): 1256–1266, 2019.

Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations, 2018.

Muzammal Naseer, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Fatih Porikli. A selfsupervised approach for adversarial robustness. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 262–271, 2020.

Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International Conference on Machine Learning, pp. 8162–8171. PMLR, 2021.

Weili Nie, Brandon Guo, Yujia Huang, Chaowei Xiao, Arash Vahdat, and Anima Anandkumar. Diffusion models for adversarial purification. In International conference on machine learning. PMLR, 2022.

Yao Qin, Nicholas Carlini, Garrison Cottrell, Ian Goodfellow, and Colin Raffel. Imperceptible, robust, and targeted adversarial examples for automatic speech recognition. In International conference on machine learning, pp. 5231–5240. PMLR, 2019.

Krishan Rajaratnam and Basemah Alshemali. Speech coding and audio preprocessing for mitigating and detecting audio adversarial examples on automatic speech recognition. http://cs.uccs. edu/˜jkalita/work/reu/REU2018/07Rajaratnam.pdf, 2018.

Mirco Ravanelli and Yoshua Bengio. Speaker recognition from raw waveform with sincnet. In 2018 IEEE Spoken Language Technology Workshop (SLT), pp. 1021–1028. IEEE, 2018.

Mirco Ravanelli, Titouan Parcollet, and Yoshua Bengio. The pytorch-kaldi speech recognition toolkit. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 6465–6469. IEEE, 2019.

Douglas A Reynolds, Thomas F Quatieri, and Robert B Dunn. Speaker verification using adapted gaussian mixture models. Digital signal processing, 10(1-3):19–41, 2000.

Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.

Hadi Salman, Mingjie Sun, Greg Yang, Ashish Kapoor, and J Zico Kolter. Denoised smoothing: A provable defense for pretrained classifiers. Advances in Neural Information Processing Systems, 33:21945–21957, 2020.

Pouya Samangouei, Maya Kabkab, and Rama Chellappa. Defense-gan: Protecting classifiers against adversarial attacks using generative models. In International Conference on Learning Representations, 2018.

Simo Sarkk ¨ a and Arno Solin. ¨ Applied Stochastic Differential Equations, volume 10. Cambridge University Press, 2019.

Ali Shafahi, Mahyar Najibi, Mohammad Amin Ghiasi, Zheng Xu, John Dickerson, Christoph Studer, Larry S Davis, Gavin Taylor, and Tom Goldstein. Adversarial training for free! Advances in Neural Information Processing Systems, 32, 2019.

Changhao Shan, Junbo Zhang, Yujun Wang, and Lei Xie. Attention-based end-to-end models for small-footprint keyword spotting. Proc. Interspeech 2018, pp. 2037–2041, 2018.

Jonathan Shen, Patrick Nguyen, Yonghui Wu, Zhifeng Chen, Mia X Chen, Ye Jia, Anjuli Kannan, Tara Sainath, Yuan Cao, Chung-Cheng Chiu, et al. Lingvo: a modular and scalable framework for sequence-to-sequence modeling. arXiv preprint arXiv:1902.08295, 2019.

Changhao Shi, Chester Holtz, and Gal Mishne. Online adversarial purification based on selfsupervised learning. In International Conference on Learning Representations, 2021.

Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In International Conference on Learning Representations, 2015.

David Snyder, Daniel Garcia-Romero, Gregory Sell, Daniel Povey, and Sanjeev Khudanpur. Xvectors: Robust dnn embeddings for speaker recognition. In 2018 IEEE international conference on acoustics, speech and signal processing (ICASSP), pp. 5329–5333. IEEE, 2018.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021a.

Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in Neural Information Processing Systems, 32, 2019.

Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. Maximum likelihood training of scorebased diffusion models. Advances in Neural Information Processing Systems, 34:1415–1428, 2021b.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021c.

Jiachen Sun, Yulong Cao, Christopher B Choy, Zhiding Yu, Anima Anandkumar, Zhuoqing Morley Mao, and Chaowei Xiao. Adversarially robust 3d point cloud recognition using self-supervisions. NeurIPS, 34:15498–15512, 2021.

Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. In International Conference on Learning Representations, 2014.

Florian Tramer and Dan Boneh. Adversarial training and robustness for multiple perturbations. Advances in Neural Information Processing Systems, 32, 2019.

Florian Tramer, Nicholas Carlini, Wieland Brendel, and Aleksander Madry. On adaptive attacks to adversarial example defenses. Advances in Neural Information Processing Systems, 33:1633– 1645, 2020.

Pete Warden. Speech commands: A dataset for limited-vocabulary speech recognition. arXiv preprint arXiv:1804.03209, 2018.

Eric Wong, Leslie Rice, and J Zico Kolter. Fast is better than free: Revisiting adversarial training. In International Conference on Learning Representations, 2020.

Chaowei Xiao, Bo Li, Jun-Yan Zhu, Warren He, Mingyan Liu, and Dawn Song. Generating adversarial examples with adversarial networks. In IJCAI, 2018a.

Chaowei Xiao, Jun-Yan Zhu, Bo Li, Warren He, Mingyan Liu, and Dawn Song. Spatially transformed adversarial examples. In International Conference on Learning Representations, 2018b.

Chaowei Xiao, Dawei Yang, Bo Li, Jia Deng, and Mingyan Liu. Meshadv: Adversarial meshes for visual recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 6898–6907, 2019.

Chaowei Xiao, Zhongzhu Chen, Kun Jin, Jiongxiao Wang, Weili Nie, Mingyan Liu, Anima Anandkumar, Bo Li, and Dawn Song. Densepure: Understanding diffusion models towards adversarial robustness. arXiv preprint arXiv:2211.00322, 2022a.

Chaowei Xiao, Xinlei Pan, Warren He, Jian Peng, Mingjie Sun, Jinfeng Yi, Mingyan Liu, Bo Li, and Dawn Song. Characterizing attacks on deep reinforcement learning. AAMAS, 2022b.

Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, and Kaiming He. Aggregated residual trans- ´ formations for deep neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1492–1500, 2017.

Zhuolin Yang, Bo Li, Pin-Yu Chen, and Dawn Song. Characterizing audio adversarial examples using temporal dependency. In International Conference on Learning Representations, 2019.

Jongmin Yoon, Sung Ju Hwang, and Juho Lee. Adversarial purification with score-based generative models. In International Conference on Machine Learning, pp. 12062–12072. PMLR, 2021.

Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. In British Machine Vision Conference 2016. British Machine Vision Association, 2016.

Dinghuai Zhang, Tianyuan Zhang, Yiping Lu, Zhanxing Zhu, and Bin Dong. You only propagate once: Accelerating adversarial training via maximal principle. Advances in Neural Information Processing Systems, 32, 2019a.

Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric Xing, Laurent El Ghaoui, and Michael Jordan. Theoretically principled trade-off between robustness and accuracy. In International conference on machine learning, pp. 7472–7482. PMLR, 2019b.

Huan Zhang, Hongge Chen, Chaowei Xiao, Bo Li, Duane Boning, and Cho-Jui Hsieh. Towards stable and efficient training of verifiably robust neural networks. ICLR, 2019c.

## APPENDIX

## A DETAILS ON TRAINING THE IMPROVE DDPM.

We train an Improved DDPM using the official repository (https://github.com/openai/improveddiffusion). For the UNet model, we set $i m a g e \_ s i z e \ = \ 3 2 ,$ num channels = 3, and num res blocks = 128. For diffusion flags, we set $N = 2 0 0 , \beta _ { 1 } = 0 . 0 0 0 1 , \beta _ { N } = 0 . 0 2$ and use the linear variance schedule. For the model training, we set the learning rate to 1e − 4 and the batch size to 230. The training loss has converged after 80,000 training steps, and we use this checkpoint to build our purifier.

## B ADDITIONAL EXPERIMENTS OF TRANSFER-BASED ATTACK

We additionally evaluate our method under transfer-based attack, where we assume the attacker can only get the output logits of the acoustic system but have no knowledge about the used defense.

We use model functional stealing to train a surrogate model. Specifically, we first feed input examples into the acoustic system consisting of DiffWave and a ResNeXt classifier and get the output logits. Then we use these output logits of the acoustic system as labels and train a new surrogate ResNeXt model, which has the same architecture as the classifier in the acoustic system. The results are shown in Table A. The Stealing Acc. denotes the accuracy of the surrogate classifier using the predictions of the defended acoustic system as ground truth. The Transfer to Vanilla and Transfer to Defended represent the undefended vanilla classifier and the defended acoustic system. The surrogate classifier is attacked to generate adversarial examples, and these adversarial examples are transferred to evaluate the robustness of the undefended vanilla classifier and the defended acoustic system.

Table A: Transfer-based attack via model functional stealing. We train a surrogate model, using the outputs of the defended acoustic system as labels. Then adversarial examples are generated by attacking the surrogate model and transferred to the undefended vanilla classifier and the defended acoustic system.
<table><tr><td>Stealing Target</td><td>Stealing Acc.</td><td colspan="2">Transfer to Vanilla Clean Robust</td><td colspan="2">Transfer to Defended Robust</td></tr><tr><td>AudioPure  $( n ^ { \star } = 1 )$ </td><td>100</td><td>100</td><td>22</td><td>Clean 100 99</td></tr><tr><td>AudioPure  $( n ^ { \star } = 5 )$ </td><td>98</td><td>100</td><td>58</td><td>96 94</td></tr></table>

## C DETAILS ABOUT CERTIFIED ROBUSTNESS

Randomized smoothing (Cohen et al., 2019) provides a provable robustness guarantee in $\mathcal { L } _ { \mathrm { { 2 } } \mathrm { { - } } \mathrm { { n o r m } } }$ by evaluating models under noise. Usually, the performance of the vanilla classifier will degrade when feeding the Gaussian perturbed inputs. To alleviate this problem, we can re-train a new network or fine-tune a pretrained network on Gaussian augmented data. However, both of them could take a lot of time on training. Another way is to apply a denoiser before the vanilla classifier, named denoised smoothing (Salman et al., 2020). Since the reverse process of the diffusion model can be seen as a good denoiser, we can use a pretrained diffusion model as a plug-and-play method to make any model certifiably robust. For a given noise level $\sigma ,$ we can compute the corresponding diffusion step $t ^ { \star }$ which adds the same level of noise to the input examples. The diffusion process can be reformulated as:

$$
\mathbf { x } _ { n } = \sqrt { \bar { \alpha } _ { n } } \mathbf { x } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \mathbf { z } = \sqrt { \bar { \alpha } _ { n } } ( \mathbf { x } _ { 0 } + \sqrt { \frac { 1 - \bar { \alpha } _ { n } } { \bar { \alpha } _ { n } } } \mathbf { z } ) , \qquad \mathbf { z } \sim \mathcal { N } ( 0 , \mathbf { I } ) ,\tag{S17}
$$

while the noisy input ˆx of randomized smoothing is

$$
\begin{array} { r } { \hat { \mathbf { x } } = \mathbf { x } _ { 0 } + \sqrt { \sigma } \mathbf { z } , \qquad \mathbf { z } \sim \mathcal { N } ( 0 , \mathbf { I } ) . } \end{array}\tag{S18}
$$

So we can obtain $n ^ { \star }$ s.t. $\begin{array} { r } { \frac { 1 - \bar { \alpha } _ { n } } { \bar { \alpha } _ { n } } = \sigma } \end{array}$ after multiplying a rescale coefficient $\sqrt { \bar { \alpha } _ { n } }$ on the input ˆx.

According to Carlini et al. (2022), a single reverse step is able to recover an image with a high accuracy for the classifier and can largely save computational time by directly recovering the data√ through $\begin{array} { r } { \mathbf { x } _ { 0 } = \frac { 1 } { \sqrt { \bar { \alpha } _ { n } } } ( \mathbf { x } _ { n } - \sqrt { 1 - \bar { \alpha } _ { n } } \mathbf { \epsilon } _ { \theta } \bar { ( } \sqrt { \bar { \alpha } _ { n } } \hat { \mathbf { x } } , n ) \mathrm { , } } \end{array}$ . So we can just apply one-shot denoising instead of running full steps in our reverse process.

Figure A shows the certified accuracy of AudioPure compared with RS-Gaussian and RS-Vanilla. The results show that the certified robustness of our method is consistently better than baselines except at small certified radii when $\sigma = 0 . 5 0$

<!-- image-->

<!-- image-->  
Figure A: Certified robustness $( \mathcal { L } _ { 2 } )$ with different input noise level $\sigma$ . Larger σ ensures better robustness under larger perturbations, but the performance for benign inputs will be degraded.

## D THEORETICAL ANALYSIS ON THE PURIFICATION ABILITY

Theorem D. 1. Assume that $p ( x )$ and $q ( x )$ are respectively the data distribution of clean examples and the data distribution of adversarial examples. We use $p _ { t }$ and $q _ { t }$ to represent the respective distribution of $x ( t )$ when $x ( t ) \sim p ( x )$ and $x ( t )$ when $x ( t ) \sim q ( x )$ . Then we have

$$
\frac { \partial D _ { K L } ( p _ { t } | | q _ { t } ) } { \partial t } \leq 0\tag{S19}
$$

where the equality is established only if $p _ { t } ~ = ~ q _ { t }$ . This inequality indicates that as t increases from 0 to 1, the KL divergence of $p _ { t }$ and $q _ { t }$ monotonically decreases. In other words, when the diffusion steps $n ^ { \star }$ increases, more of the adversarial perturbations will be removed. Considering that the original semantic information will also be removed if $n ^ { \star }$ is too large, which affects the clean accuracy, there should be a trade-off when we set $n ^ { \star }$ for the diffusion model purifier.

Proof: Following Nie et al. (2022); Song et al. (2021b), we firstly formulate the Fokker-Planck equation (Sarkk ¨ a & Solin, 2019) of the forward SDE in Eq. 7 (where we define ¨ $\begin{array} { r } { f ( x , t ) : = - \frac { 1 } { 2 } \beta ( t ) } \end{array}$ and $g ( t ) : = \sqrt { \beta ( t ) } )$ as:

$$
\begin{array} { r l } & { \displaystyle \frac { \partial p _ { t } ( x ) } { \partial t } = - \nabla _ { x } \bigg ( f ( x , t ) p _ { t } ( x ) - \frac { 1 } { 2 } g ^ { 2 } ( t ) \nabla _ { x } p _ { t } ( x ) \bigg ) } \\ & { \quad \quad \quad = - \nabla _ { x } \bigg ( f ( x , t ) p _ { t } ( x ) - \frac { 1 } { 2 } g ^ { 2 } ( t ) p _ { t } ( x ) \nabla _ { x } \log p _ { t } ( x ) \bigg ) } \\ & { \quad \quad = \nabla _ { x } \cdot ( h _ { p } ( x , t ) p _ { t } ( x ) ) } \end{array}\tag{S20}
$$

where $h _ { p } ( x , t ) : = \textstyle { \frac { 1 } { 2 } } g ^ { 2 } ( t ) \nabla _ { x } \log p _ { t } ( x ) - f ( x , t )$ . Assuming pt and $q _ { t }$ are smooth and fast decaying, i.e. for any $i = 1 , \ldots , d ,$ we have

$$
\operatorname * { l i m } _ { x _ { i }  \infty } p _ { t } ( x ) \frac { \partial } { \partial x _ { i } } \log p _ { t } ( x ) = 0 , \qquad \operatorname * { l i m } _ { x _ { i }  \infty } q _ { t } ( x ) \frac { \partial } { \partial x _ { i } } \log q _ { t } ( x ) = 0\tag{S21}
$$

for $x _ { i } .$ , the i-th dimension of $x \in \mathbb { R } ^ { d }$ . Then we reformulate the KL divergence as

$$
\begin{array} { r l } & { \frac { \partial D _ { K L } ( p _ { t } | | q _ { t } ) } { \partial t } = - \frac { \partial } { \partial t } \int _ { \mathcal { P } _ { t } ( x ) } \log \frac { p _ { t } ( x ) } { \mathcal { H } ( x ) } \mathrm { d } x } \\ & { \qquad = - \nabla _ { x } \left( f ( x , t ) p _ { t } ( x ) - \frac { 1 } { 2 } g ^ { 2 } ( t ) p _ { t } ( x ) \nabla _ { x } \log p _ { t } ( x ) \right) } \\ & { \qquad = \int \nabla _ { x } \cdot ( h _ { p } ( x , t ) p _ { t } ( x ) ) \log \frac { p _ { t } ( x ) } { \mathcal { H } ( x ) } \mathrm { d } x + \int \frac { p _ { t } ( x ) } { \mathcal { H } ( x ) } \nabla _ { x } \cdot ( h _ { p } ( x , t ) p _ { t } ( x ) ) \mathrm { d } x } \\ & { \qquad = - \int p _ { t } ( x ) [ h _ { p } ( x , t ) - h _ { q } ( x , t ) ] ^ { \top } [ \nabla _ { x } \log p _ { t } ( x ) - \nabla _ { x } \log q _ { t } ( x ) ] \mathrm { d } x } \\ & { \qquad = - \frac { 1 } { 2 } g ^ { 2 } ( t ) \int p _ { t } ( x ) \| \nabla _ { x } \log p _ { t } ( x ) - \nabla _ { x } \log q _ { t } ( x ) \| _ { 2 } ^ { 2 } \mathrm { d } x } \\ & { \qquad = - \frac { 1 } { 2 } g ^ { 2 } ( t ) D _ { F } ( p _ { t } | | q _ { t } ) } \end{array}\tag{S22}
$$

where $D _ { F } ( p _ { t } | | q _ { t } )$ is the Fisher divergence. Considering that $g ^ { 2 } ( t ) = \beta ( t ) > 0$ , and the Fisher divergence $D _ { F } ( p _ { t } | | q _ { t } ) \geq 0$ and the equality is established only if $p _ { t } = q _ { t }$ , as a result, we have Eq S19, where the equality is established only if $p _ { t } = q _ { t }$ •

## E EXPERIMENTS ON THE QUALCOMM KEYWORD SPEECH DATASET

In addition to the commonly used SC09, for a more comprehensive consideration, we also conduct experiments on the Qualcomm Keyword Speech Dataset (Kim et al., 2019), denoted as QKW in the following. QKW consists of 4270 utterances belonging to four classes, with variable durations from 0.48s to 1.92s. We split them into a training set (3770 utterances), a validation set (400 utterances), and a test set (100 utterances). To handle the variable-sized input, we train an Attention Recurrent Convolutional Network (Shan et al., 2018) and save the checkpoint with the highest accuracy on the validation set. Then we finetuned the DiffWave model on QKW for 50,000 steps, with $l r = 2 e - 4$ and batch size per gpu = 2 for 3 GPU. The results under $\mathcal { L } _ { \infty } \mathrm { P G D } _ { 1 0 }$ with $\epsilon = 0 . 0 0 2$ are shown in Table B. We can observe that AudioPure can still achieve non-trivial robustness and handle the audio with variable time duration well.

Table B: We apply AudioPure to the Qualcomm Keyword Speech Dataset. The diffusion steps $n ^ { \star }$ set to 2.
<table><tr><td>Defense</td><td>Clean</td><td>Robust</td></tr><tr><td>None</td><td>100</td><td>0</td></tr><tr><td>AudioPure</td><td>91</td><td>61</td></tr></table>

## F FINE-TUNING ON ADVERSARIAL EXAMPLES

AudioPure takes advantage of pretrained diffusion models. We wonder whether the purification performance will be improved if fine-tuned on adversarial examples. And we further fine-tune the DiffWave model by augmenting self-supervised perturbation (SSP) (Naseer et al., 2020). Specifically, we use STFT (rescaling to the Mel-scale) as our feature extractor and maximize the following objective to generate perturbed examples:

$$
\underset { x ^ { \prime } } { \arg \operatorname* { m a x } } \Delta ( x , x ^ { \prime } ) = \| S T F T ( x ) , S T F T ( x ^ { \prime } ) \| _ { \infty } , \quad s . t . \| x - x ^ { \prime } \| _ { \infty }\tag{S23}
$$

where x is the clean example and $x ^ { \prime }$ is the perturbed example. We then use gradient descent to optimize the perturbed example by:

$$
x _ { t + 1 } ^ { \prime } = c l i p ( x _ { t } ^ { \prime } + \alpha \cdot s i g n ( \nabla _ { x } \Delta ( x , x _ { t } ^ { \prime } ) ) , x - \epsilon , x + \epsilon ) ,\tag{S24}
$$

for $t = 1 , \dots , T$ Here we use $T = 1 0 0 , \epsilon = 0 . 0 0 2$ and $\alpha = 0 . 0 0 0 4$ . Next, we fine-tune the pretrained DiffWave model on the SSP examples, minimizing the following loss:

$$
\begin{array} { r } { \mathcal { L } _ { t u n i n g } = \mathcal { L } _ { a u d i o } + \lambda \mathcal { L } _ { f e a t } , } \end{array}\tag{S25}
$$

where

$$
\mathcal { L } _ { a u d i o } = M S E ( x , \mathbf { P u r i f i e r } ( x _ { t } ^ { \prime } , n ^ { \star } ) ) ,\tag{S26}
$$

$$
\mathcal { L } _ { f e a t } = M S E ( S T F T ( x ) , S T F T ( \mathbf { P u r i f i e r } ( x _ { t } ^ { \prime } , n ^ { \star } ) ) .\tag{S27}
$$

We choose $\lambda = 0 . 1$ 1 and use SGD to optimize $\mathcal { L } _ { t u n i n g } ,$ setting the learning rate to $1 e - 5$ . The results are shown in Table C. As a result, it does not improve the performance of AudioPure (with $n ^ { \star } = 3 )$ under $\mathcal { L } _ { \infty } \mathrm { P G D } _ { 1 0 }$ and $\mathrm { P G D } _ { 7 0 }$ with $\epsilon = 0 . 0 0 2$ . These results further verify the effectiveness of using pretrained models.

Table C: We fine-tune the pretrained DiffWave model on adversarial examples generated by SSP. After fine-tuning, the performance is not improved.
<table><tr><td>Defense</td><td>Clean</td><td> $\mathrm { P G D _ { 1 0 } }$ </td><td> $\mathrm { P G D } _ { 7 0 }$ </td></tr><tr><td>None</td><td>100</td><td>3</td><td>1</td></tr><tr><td>AudioPure</td><td>97</td><td>89</td><td>84</td></tr><tr><td>SSP-Tuned  $\mathsf { A u d i o P u r e }$ </td><td>97</td><td>89</td><td>82</td></tr></table>

## G COMPARISON WITH OTHER DENOISER-BASED DEFENSE

We compare AudioPure with DefenseGAN (Samangouei et al., 2018) and Joint Adversarial Finetuning (Joshi et al., 2022). For DefenseGAN, which is originally designed to defend against adversarial images by finding the optimal noise that generates the most similar image to the adversarial counterpart, we adopt it to the audio domain, choosing WaveGAN (Donahue et al., 2018) as the GAN model in this pipeline. We train a WaveGAN on the SC09 dataset for 100 epochs, using the Adam optimizer with $l r = 1 e - 3 , \beta _ { 1 } = 0 . 5 ,$ and $\beta _ { 2 } = 0 . 9$ . For Joint Adversarial Fine-tuning, we follow the setting of Joshi et al. (2022), using a Conv-TasNet (Luo & Mesgarani, 2019) as the denoiser. And like Joshi et al. (2022), we craft an offline adversarial SC09 dataset against the pretrained classifier by using L-inf PGD-100 attacks with $\epsilon = 0 . 0 0 2$ (denoted as OffAdv-SC09). Then we train a Conv-TasNet model on OffAdv-SC09 for 30 epochs to get the pretrained denoiser. We denote the defense using the pretrained Conv-TasNet as CTN Baseline. Based on the adversarial examples generated by attacking the whole acoustic system, we only update the Conv-TasNet denoiser while keeping the classifier frozen, and denote this method as CTN Adv-Finetune-Joint-frozen. During the adversarial tuning, we use $\mathcal { L } _ { \infty } \mathrm { { P G D } _ { 1 0 } }$ attack with $\epsilon = 0 . 0 0 2$ . After tuning for 1000 steps with batch $s i z e = 2 0$ , we calculate the clean and robust accuracy (under $\mathcal { L } _ { \infty } \mathrm { { P G D } _ { 1 0 } }$ and $\mathrm { P G D } _ { 7 0 }$ with $\epsilon = 0 . 0 0 2 )$ on the same test used in our paper.

We report the results in Table D. We find that DefenseGAN based on WaveGAN cannot work well in the audio domain. It shows the impact of domain differences with respect to the final results and verifies the importance of our pipeline design. Besides, the Conv-TasNet denoiser is less effective than diffusion models against adaptive attacks, even after fine-tuning.

Table D: We compare AudioPure with different denoiser-based defenses. DiffWave is proven to be a more effective purifier.
<table><tr><td>Defense</td><td>Clean</td><td> $\mathrm { P G D _ { 1 0 } }$ </td><td> $\mathrm { P G D } _ { 7 0 }$ </td></tr><tr><td>None</td><td>100</td><td>3</td><td>1</td></tr><tr><td>AudioPure</td><td>97</td><td>89</td><td>84</td></tr><tr><td>DefenseGAN</td><td>8</td><td>0</td><td>0</td></tr><tr><td>CTN Baseline</td><td>98</td><td>13</td><td>1</td></tr><tr><td>CTN Adv-Finetune-Joint-frozen</td><td>90</td><td>52</td><td>41</td></tr></table>

## H COMPARISON WITH THE REGULARIZATION-BASED DEFENSE

Gu & Rigazio (2014); Hoffman et al. (2019) introduce the input-output Jacobian matrix of the network as a regularization term in the optimization objective, formulated as

$$
\mathcal { L } _ { r e g } = \sum _ { i } \left( \mathcal { L } ( x _ { i } , y _ { i } ) + \lambda \| \frac { \partial f ( x _ { i } ) } { \partial x _ { i } } \| _ { F } \right) ,\tag{S28}
$$

where $x _ { i } \in \mathbb { R } ^ { d }$ is the input data, $y _ { i } \in \mathbb { R } ^ { n }$ is the label, $\mathcal { L } : \mathbb { R } ^ { d } \times \mathbb { R } ^ { n }  \mathbf { \Phi } ^ { \cdot }$ R is the original loss function, and $f : \mathbb { R } ^ { d }  \mathbb { R } ^ { n }$ is the neural network. By minimizing the Frobenius norm of the Jacobian matrix, the adversarial robustness of the network will be improved. For a more comprehensive study, we also compare AudioPure with this regularization-based method, using different λ. The results are shown in Table E, where we denote the regularization-based defense as Jacobian-Reg.

Table E: We compare AudioPure with the regularization-based defense, using different λ.
<table><tr><td>Defense</td><td>Clean</td><td> $\mathrm { P G D _ { 1 0 } }$ </td><td> $\mathrm { P G D } _ { 7 0 }$ </td></tr><tr><td>None</td><td>100</td><td>3</td><td>1</td></tr><tr><td>AudioPure</td><td>97</td><td>89</td><td>84</td></tr><tr><td>Jacobian-Reg (λ=1e-8)</td><td>45</td><td>9</td><td>5</td></tr><tr><td>Jacobian-Reg (λ=1e-9)</td><td>84</td><td>27</td><td>15</td></tr><tr><td>Jacobian-Reg (λ=1e-10)</td><td>91</td><td>31</td><td>18</td></tr><tr><td>Jacobian-Reg (λ=1e-11)</td><td>96</td><td>19</td><td>4</td></tr></table>

## I EXPERIMENTS ON LARGER ATTACK BUDGETS.

Besides the results of different ϵ in Table 2, we conduct additional experiments to explore the potential of the diffusion model for purification. We select $\epsilon = \{ 0 . 0 1 , 0 . \bar { 0 } 2 , 0 . 0 3 , 0 . 0 4 , \bar { 0 . 0 5 } \}$ , and set the diffusion steps $n ^ { \star }$ . The results are shown in Table F. We find that our method still achieves 42% accuracy at $\epsilon = 0 . 0 3$ , which brings significant distortions to audio. Our method keeps the ability to purify adversarial perturbations until $\epsilon = 0 . 0 5$ . We also visualize the audio waveforms under attacks with different ϵ, illustrated in Figure B. It is easy to observe significant noise in them.

Table F: We explore the potential of DiffWave under larger attack budgets. The diffusion steps $n ^ { \star }$ set to 5.
<table><tr><td>Attack Budget</td><td> $\epsilon = 0 . 0 1$ </td><td> $\epsilon = 0 . 0 2$   $\epsilon = 0 . 0 3$ </td><td> $\epsilon = 0 . 0 4$ </td><td> $\epsilon = 0 . 0 5$ </td></tr><tr><td>Robust Acc.</td><td>82</td><td>67 42</td><td>14</td><td>0</td></tr></table>

## J ADDITIONAL INFERENCE TIME COST.

Due to the introduction of diffusion models, AudioPure will bring additional time cost during inference. As shown in Table G, we compute the time cost per audio, averaged on 100 examples and the time duration for each example is around one second. We evaluate it on an NVIDIA RTX 3090 GPU with Intel® Core™ i9-10920X CPU @ 3.50GHz and 64 GB RAM.

Table G: The inference time cost when using different diffusion steps $n ^ { \star }$
<table><tr><td>Diffusion Steps</td><td> $n ^ { \star } = 0$ </td><td> $n ^ { \star } = 1$ </td><td> $n ^ { \star } = 2$ </td><td> $n ^ { \star } = 3$ </td><td> $n ^ { \star } = 5$ </td><td> $n ^ { \star } = 7$ </td><td> $n ^ { \star } = 1 0$ </td></tr><tr><td>Time Cost (s)</td><td>0.0967</td><td>0.5522</td><td>0.7876</td><td>1.0162</td><td>1.4795</td><td>2.0125</td><td>2.6839</td></tr></table>

<!-- image-->  
(a)

<!-- image-->  
(b)

<!-- image-->  
(c)  
Figure B: Visualizations of the clean audio and adversarial audio with different attack budgets.



<!-- image-->  
(d)

<!-- image-->  
(e)

<!-- image-->  
(f)  
Figure B: Visualizations of the clean audio and adversarial audio with different attack budgets.



