# DiffAttack: Evasion Attacks Against Diffusion-Based Adversarial Purification

Mintong Kang UIUC mintong2@illinois.edu

Dawn Song UC Berkeley dawnsong@berkeley.edu

Bo Li UIUC lbo@illinois.edu

# Abstract

Diffusion-based purification defenses leverage diffusion models to remove crafted perturbations of adversarial examples and achieve state-of-the-art robustness. Recent studies show that even advanced attacks cannot break such defenses effectively, since the purification process induces an extremely deep computational graph which poses the potential problem of vanishing/exploding gradient, high memory cost, and unbounded randomness. In this paper, we propose an attack technique DiffAttack to perform effective and efficient attacks against diffusion-based purification defenses, including both DDPM and score-based approaches. In particular, we propose a deviated-reconstruction loss at intermediate diffusion steps to induce inaccurate density gradient estimation to tackle the problem of vanishing/exploding gradients. We also provide a segment-wise forwarding-backwarding algorithm, which leads to memory-efficient gradient backpropagation. We validate the attack effectiveness of DiffAttack compared with existing adaptive attacks on CIFAR-10 and ImageNet. We show that DiffAttack decreases the robust accuracy of models compared with SOTA attacks by over $20 \%$ on CIFAR-10 under $\ell _ { \infty }$ attack $\overset { \prime } { \epsilon } = 8 / 2 5 5 \overset { \mathrm { ~ , ~ } } { \delta }$ , and over $10 \%$ on ImageNet under $\ell _ { \infty }$ attack $\acute { \epsilon } = 4 / 2 5 5 \acute { ) }$ ). We conduct a series of ablations studies, and we find 1) DiffAttack with the deviated-reconstruction loss added over uniformly sampled time steps is more effective than that added over only initial/final steps, and 2) diffusion-based purification with a moderate diffusion length is more robust under DiffAttack.

# 1 Introduction

Since deep neural networks (DNNs) are found vulnerable to adversarial perturbations [52, 20], improving the robustness of neural networks against such crafted perturbations has become important, especially in safety-critical applications [18, 5, 54]. In recent years, many defenses have been proposed, but they are attacked again by more advanced adaptive attacks [7, 30, 11, 12]. One recent line of defense (diffusion-based purification) leverages diffusion models to purify the input images and achieves the state-of-the-art robustness. Based on the type of diffusion models the defense utilizes, diffusion-based purification can be categorized into score-based purification [34] which uses the score-based diffusion model [49] and DDPM-based purification [4, 62, 57, 51, 55, 56] which uses the denoising diffusion probabilistic model (DDPM) [25]. Recent studies show that even the most advanced attacks [12, 34] cannot break these defenses due to the challenges of vanishing/exploding gradient, high memory cost, and large randomness. In this paper, we aim to explore the vulnerabilities of such diffusion-based purification defenses, and design a more effective and efficient adaptive attack against diffusion-based purification, which will help to better understand the properties of diffusion process and motivate future defenses.

In particular, the diffusion-based purification defenses utilize diffusion models to first diffuse the adversarial examples with Gaussian noises and then perform sampling to remove the noises. In this way, the hope is that the crafted adversarial perturbations can also be removed since the training distribution of diffusion models is clean [49, 25]. The diffusion length (i.e., the total diffusion time steps) is usually large, and at each time step, the deep neural network is used to estimate the gradient of the data distribution. This results in an extremely deep computational graph that poses great challenges of attacking it: vanishing/exploding gradients, unavailable memory cost, and large randomness. To tackle these challenges, we propose a deviated-reconstruction loss and a segment-wise forwarding-backwarding algorithm and integrate them as an effective and efficient attack technique DiffAttack.

Essentially, our deviated-reconstruction loss pushes the reconstructed samples away from the diffused samples at corresponding time steps. It is added at multiple intermediate time steps to relieve the problem of vanishing/exploding gradients. We also theoretically analyze the connection between it and the score-matching loss [26], and we prove that maximizing the deviated-reconstruction loss induces inaccurate estimation of the density gradient of the data distribution, leading to a higher chance of attacks. To overcome the problem of large memory cost, we propose a segment-wise forwarding-backwarding algorithm to backpropagate the gradients through a long path. Concretely, we first do a forward pass and store intermediate samples, and then iteratively simulate the forward pass of a segment and backward the gradient following the chain rule. Ignoring the memory cost induced by storing samples (small compared with the computational graph), our approach achieves $\mathcal { O } ( 1 )$ memory cost.

Finally, we integrate the deviated-reconstruction loss and segment-wise forwarding-backwarding algorithm into DiffAttack, and empirically validate its effectiveness on CIFAR-10 and ImageNet. We find that (1) DiffAttack outperforms existing attack methods [34, 60, 53, 1, 2] by a large margin for both the score-based purification and DDPM-based purification defenses, especially under large perturbation radii; (2) the memory cost of our efficient segment-wise forwarding-backwarding algorithm does not scale up with the diffusion length and saves more than $1 0 \mathrm { x }$ memory cost compared with the baseline [4]; (3) a moderate diffusion length benefits the robustness of the diffusion-based purification since longer length will hurt the benign accuracy while shorter length makes it easier to be attacked; (4) attacks with the deviated-reconstruction loss added over uniformly sampled time steps outperform that added over only initial/final time steps. The effectiveness of DiffAttack and interesting findings will motivate us to better understand and rethink the robustness of diffusion-based purification defenses.

We summarize the main technical contributions as follows:

• We propose DiffAttack, a strong evasion attack against the diffusion-based adversarial purification defenses, including score-based and DDPM-based purification.   
We propose a deviated-reconstruction loss to tackle the problem of vanishing/exploding gradient, and theoretically analyze its connection with data density estimation.   
We propose a segment-wise forwarding-backwarding algorithm to tackle the high memory cost challenge, and we are the first to adaptively attack the DDPM-based purification defense, which is hard to attack due to the high memory cost. We empirically demonstrate that DiffAttack outperforms existing attacks by a large margin on CIFAR-10 and ImageNet. Particularly, DiffAttack decreases the model robust accuracy by over $2 0 \%$ for $\ell _ { \infty }$ attack $\acute { \epsilon } = 8 / 2 5 5$ ) on CIFAR-10, and over $10 \%$ on ImageNet under $\ell _ { \infty }$ attack $\acute { \epsilon } = 4 / 2 5 5$ ). We conduct a series of ablation studies and show that (1) a moderate diffusion length benefits the model robustness, and (2) attacks with the deviated-reconstruction loss added over uniformly sampled time steps outperform that added over only initial/final time steps.

# 2 Preliminary

There are two types of diffusion-based purification defenses, DDPM-based purification, and scorebased purification, which leverage DDPM [46, 25] and score-based diffusion model [49] to purify the adversarial examples, respectively. Next, we will introduce the basic concepts of DDPM and score-based diffusion models.

Denote the diffusion process indexed by time step $t$ with the diffusion length $T$ by $\{ \mathbf { x } _ { t } \} _ { t = 0 } ^ { T }$ DDPM constructs a discrete Markov chain $\{ \mathbf { x } _ { t } \} _ { t = 0 } ^ { \hat { T } }$ with discrete time variables $t$ following $p ( \mathbf { x } _ { t } | \mathbf { x } _ { t - 1 } ) \ = \ { \mathcal { N } } ( \mathbf { x } _ { t } ; { \sqrt { 1 - \beta _ { t } } } \mathbf { x } _ { t - 1 } , \beta _ { t } \mathbf { I } )$ where $\beta _ { t }$ is a sequence of positive noise scales (e.g., linear scheduling, cosine scheduling [33]). Considering $\alpha _ { t } ~ : = ~ 1 - \beta _ { t }$ , $\bar { \alpha } _ { t } : = \Pi _ { s = 1 } ^ { t } \alpha _ { s }$ , and $\sigma _ { t } = \sqrt { \beta _ { t } ( 1 - \bar { \alpha } _ { t - 1 } ) / ( 1 - \bar { \alpha } _ { t } } )$ , the reverse process (i.e., sampling process) can be formulated as:

$$
\mathbf { x } _ { t - 1 } = \frac { 1 } { \sqrt { \alpha _ { t } } } \left( \mathbf { x } _ { t } - \frac { 1 - \alpha _ { t } } { \sqrt { 1 - \bar { \alpha } _ { t } } } \epsilon _ { \theta } ( \mathbf { x } _ { t } , t ) \right) + \sigma _ { t } \mathbf { z }
$$

where $\mathbf { z }$ is drawn from $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ . $\epsilon _ { \theta }$ parameterized with $\theta$ is the model to approximate the perturbation $\epsilon$ in the diffusion process and is trained via the density gradient loss $\mathcal { L } _ { d }$ :

$$
\mathcal { L } _ { d } = \mathbb { E } _ { t , \epsilon } \left[ \frac { \beta _ { t } ^ { 2 } } { 2 \sigma _ { t } ^ { 2 } \alpha _ { t } \left( 1 - \bar { \alpha } _ { t } \right) } \| \epsilon - \epsilon _ { \theta } ( \sqrt { \bar { \alpha } _ { t } } \mathbf { x } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon , t ) \| _ { 2 } ^ { 2 } \right]
$$

where $\epsilon$ is drawn from $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ and $t$ is uniformly sampled from $[ T ] : = \{ 1 , 2 , . . . , T \}$ .

Score-based diffusion model formulates diffusion models with stochastic differential equations (SDE). The diffusion process $\{ \mathbf { x } _ { t } \} _ { t = 0 } ^ { T }$ is indexed by a continuous time variable $t \in [ 0 , 1 ]$ . The diffusion process can be formulated as:

$$
d \mathbf { x } = f ( \mathbf { x } , t ) d t + g ( t ) d \mathbf { w }
$$

where $f ( \mathbf { x } , t ) : \mathbb { R } ^ { n } \mapsto \mathbb { R } ^ { n }$ is the drift coefficient characterizing the shift of the distribution, $g ( t )$ is the diffusion coefficient controlling the noise scales, and w is the standard Wiener process. The reverse process is characterized via the reverse time SDE of Equation (3):

$$
d \mathbf { x } = [ f ( \mathbf { x } , t ) - g ( t ) ^ { 2 } \nabla _ { \mathbf { x } } \log p _ { t } ( \mathbf { x } ) ] d t + g ( t ) d \mathbf { w }
$$

where $\nabla _ { \mathbf { x } } \log p _ { t } ( \mathbf { x } )$ is the time-dependent score function that can be approximated with neural networks $\mathbf { S } \theta$ parameterized with $\theta$ , which is trained via the score matching loss $\mathcal { L } _ { s }$ [26, 47]:

$$
\mathcal { L } _ { s } = \mathbb { E } _ { t } \left[ \lambda ( t ) \mathbb { E } _ { { \mathbf { x } } _ { t } \mid { \mathbf { x } } _ { 0 } } \| { \mathbf { s } } _ { \theta } ( { \mathbf { x } } _ { t } , t ) - \nabla _ { { \mathbf { x } } _ { t } } \log ( p ( { \mathbf { x } } _ { t } | { \mathbf { x } } _ { 0 } ) ) \| _ { 2 } ^ { 2 } \right]
$$

where $\lambda : [ 0 , 1 ] \to \mathbb { R }$ is a weighting function and $t$ is uniformly sampled over $[ 0 , 1 ]$

# 3 DiffAttack

# 3.1 Evasion attacks against diffusion-based purification

A class of defenses leverages generative models for adversarial purification [43, 48, 45, 60]. The adversarial images are transformed into latent representations, and then the purified images are sampled starting from the latent space using the generative models. The process is expected to remove the crafted perturbations since the training distribution of generative models is assumed to be clean. With diffusion models showing the power of image generation recently [15, 39], diffusion-based adversarial purification has achieved SOTA defense performance [34, 4].

We first formulate the problem of evasion attacks against diffusion-based purification defenses. Suppose that the process of diffusion-based purification, including the diffusion and reverse process, is denoted by $P : \mathbb { R } ^ { n } \mapsto \mathbb { R } ^ { n }$ where $n$ is the dimension of the input $\mathbf { x } _ { \mathrm { 0 } }$ , and the classifier is denoted by $F : \mathbb { R } ^ { n } \mapsto [ K ]$ where $K$ is the number of classes. Given an input pair $\left( \mathbf { x } _ { 0 } , y \right)$ , the adversarial example $\tilde { \mathbf { x } } _ { 0 }$ satisfies:

$$
\arg \operatorname* { m a x } _ { i \in [ K ] } F _ { i } ( P ( \tilde { \mathbf { x } } _ { 0 } ) ) \neq y \quad s . t . d ( \mathbf { x } _ { 0 } , \tilde { \mathbf { x } } _ { 0 } ) \leq \delta _ { m a x }
$$

where $F _ { i } ( \cdot )$ is the $i$ -th element of the output, $d : \mathbb { R } ^ { n } \times \mathbb { R } ^ { n } \mapsto \mathbb { R }$ is the distance function in the input space, and $\delta _ { m a x }$ is the perturbation budget.

Since directly searching for the adversarial instance $\tilde { \mathbf { x } } _ { 0 }$ based on Equation (6) is challenging, we often use a surrogate loss $\mathcal { L }$ to solve an optimization problem:

$$
\begin{array} { r l } { \underset { { \bf \tilde { x } } _ { 0 } } { \operatorname* { m a x } } \mathcal { L } \big ( F \big ( P ( { \tilde { \bf x } } _ { 0 } ) \big ) , y \big ) } & { { } s . t . d ( { \bf x } _ { 0 } , { \tilde { \bf x } } _ { 0 } ) \leq \delta _ { m a x } } \end{array}
$$

where $P ( \cdot )$ is the purification process with DDPM (Equation (1)) or score-based diffusion (Equations (3) and (4)), and the surrogate loss $\mathcal { L }$ is often selected as the classification-guided loss, such as CW loss [7], Cross-Entropy loss and difference of logits ratio (DLR) loss [12]. Existing adaptive attack methods such as PGD [30] and APGD attack [12] approximately solve the optimization problem in Equation (7) via computing the gradients of loss $\mathcal { L }$ with respect to the decision variable $\tilde { \mathbf { x } } _ { 0 }$ and iteratively updating $\tilde { \mathbf { x } } _ { 0 }$ with the gradients.

![](images/8babd6d8b44898ad88faa5095676e8d23e622bb4fbc3cd75ecc4192ca77fbc53.jpg)  
Figure 1: DiffAttack against diffusion-based adversarial purification defenses. DiffAttack features the deviated-reconstruction loss that addresses vanishing/exploding gradients and the segment-wise forwarding-backwarding algorithm that leads to memory-efficient gradient backpropagation.

However, we observe that the gradient computation for the diffusion-based purification process is challenging for three reasons: 1) the long sampling process of the diffusion model induces an extremely deep computational graph which poses the problem of vanishing/exploding gradient [2], 2) the deep computational graph impedes gradient backpropagation, which requires high memory cost [60, 4], and 3) the diffusion and sampling process introduces large randomness which makes the calculated gradients unstable and noisy.

To address these challenges, we propose a deviated-reconstruction loss (in Section 3.2) and a segmentwise forwarding-backwarding algorithm (in Section 3.3) and design an effective algorithm DiffAttack by integrating them into the attack technique (in Section 3.4).

# 3.2 Deviated-reconstruction loss

In general, the surrogate loss $\mathcal { L }$ in Equation (7) is selected as the classification-guided loss, such as CW loss, Cross-Entropy loss, or DLR loss. However, these losses can only be imposed at the classification layer, and induce the problem of vanishing/exploding gradients [2] due to the long diffusion length. Specifically, the diffusion purification process induces an extremely deep graph. For example, DiffPure applies hundreds of iterations of sampling and uses deep UNet with tens of layers as score estimators. Thus, the computational graph consists of thousands of layers, which could cause the problem of gradient vanishing/exploding. Similar gradient problems are also mentioned with generic score-based generative purification (Section 4, 5.1 in [60]). Backward path differentiable approximation (BPDA) attack [2] is usually adopted to overcome such problems, but the surrogate model of the complicated sampling process is hard to find, and a simple identity mapping function is demonstrated to be ineffective in the case [34, 4, 60].

To overcome the problem of exploding/vanishing gradients, we attempt to impose intermediate guidance during the attack. It is possible to build a set of classifiers on the intermediate samples in the reverse process and use the weighted average of the classification-guided loss at multiple layers as the surrogate loss $\mathcal { L }$ . However, we observe that the intermediate samples are noisy, and thus using classifier $F$ that is trained on clean data cannot provide effective gradients. One solution is to train a set of classifiers with different noise scales and apply them to intermediate samples to impose classification-guided loss, but the training is too expensive considering the large diffusion length and variant noise scales at different time steps. Thus, we propose a deviated-reconstruction loss to address the challenge via imposing discrepancy for samples between the diffusion and reverse processes adversarially to provide effective loss at intermediate time steps.

Concretely, since a sequence of samples is generated in the diffusion and reverse processes, effective loss imposed on them would relieve the problem of vanishing/exploding gradient and benefit the optimization. More formally, let $\mathbf { x } _ { t }$ , $\mathbf { x } _ { t } ^ { \prime }$ be the samples at time step $t$ in the diffusion process and the reverse process, respectively. Formally, we maximize the deviated-reconstruction loss $\mathcal { L } _ { d e v }$

formulated as follows:

$$
\begin{array} { r } { \operatorname* { m a x } \mathcal { L } _ { d e v } = \mathbb { E } _ { t } [ \alpha ( t ) \mathbb { E } _ { \mathbf { x } _ { t } , \mathbf { x } _ { t } ^ { \prime } | \mathbf { x } _ { 0 } } d ( \mathbf { x } _ { t } , \mathbf { x } _ { t } ^ { \prime } ) ] } \end{array}
$$

where $\alpha ( \cdot )$ is time-dependent weight coefficients and $d ( \mathbf { x } _ { t } , \mathbf { x } _ { t } ^ { \prime } )$ is the distance between noisy image $\mathbf { x } _ { t }$ in the diffusion process and corresponding sampled image $\mathbf { x } _ { t } ^ { \prime }$ in the reverse process. The expectation over $t$ is approximated by taking the average of results at uniformly sampled time steps in $[ 0 , T ]$ , and the loss at shallow layers in the computational graph (i.e., large time step $t$ ) helps relieve the problem of vanishing/exploding gradient. The conditional expectation over $\mathbf { x } _ { t } , \mathbf { x } _ { t } ^ { \prime }$ given $\mathbf { x } _ { \mathrm { 0 } }$ is approximated by purifying $\mathbf { x } _ { \mathrm { 0 } }$ multiple times and taking the average of the loss.

Intuitively, the deviated-reconstruction loss in Equation (8) pushes the reconstructed sample $\mathbf { x } _ { t } ^ { \prime }$ in the reverse process away from the sample $\mathbf { x } _ { t }$ at the corresponding time step in the diffusion process, and finally induces an inaccurate reconstruction of the clean image. Letting $q _ { t } ( \mathbf { x } )$ and $q _ { t } ^ { \prime } ( \mathbf { x } )$ be the distribution of $\mathbf { x } _ { t }$ and $\mathbf { x } _ { t } ^ { \prime }$ , we can theoretically prove that the distribution distance between $q _ { t } ( \mathbf { x } )$ and $q _ { t } ^ { \prime } ( \mathbf { x } )$ positively correlates with the score-matching loss of the score-based diffusion or the density gradient loss of the DDPM. In other words, maximizing the deviated-reconstruction loss in Equation (8) induces inaccurate data density estimation, which results in the discrepancy between the sampled distribution and the clean training distribution.

Theorem 1. Consider adversarial sample $\tilde { \mathbf { x } } _ { 0 } : = \mathbf { x } _ { 0 } + \delta$ , where $\mathbf { x } _ { \mathrm { 0 } }$ is the clean example and $\delta$ is the perturbation. $p _ { t } ( \mathbf { x } ) , p _ { t } ^ { \prime } ( \mathbf { x } ) , q _ { t } ( \mathbf { x } ) , q _ { t } ^ { \prime } ( \mathbf { \dot { x } } )$ are the distribution of $\mathbf { x } _ { t } , \mathbf { x } _ { t } ^ { \prime } , \tilde { \mathbf { x } } _ { t } , \tilde { \mathbf { x } } _ { t } ^ { \prime }$ where $\mathbf { x } _ { t } ^ { \prime }$ represents the reconstruction of $\mathbf { x } _ { t }$ in the reverse process. $D _ { T V } ( \cdot , \cdot )$ measures the total variation distance. Given a $V P$ -SDE parameterized by $\beta ( \cdot )$ and the score-based model $\scriptstyle { s _ { \theta } }$ with mild assumptions that $\| \nabla _ { \mathbf { x } } \log p _ { t } ( \mathbf { x } ) - s _ { \theta } ( \mathbf { x } , t ) \| _ { 2 } ^ { 2 } \leq L _ { u } , D _ { T V } ( p _ { t } , p _ { t } ^ { \prime } ) \leq \epsilon _ { r e }$ , and a bounded score function by $M$ (specified in Appendix C.1), we have:

$$
\begin{array} { l } { { \displaystyle D _ { T V } ( q _ { t } , q _ { t } ^ { \prime } ) \leq \frac { 1 } { 2 } \sqrt { \mathbb { E } _ { t , \mathbf { x } | \mathbf { x } _ { 0 } } \| s _ { \theta } ( \mathbf { x } , t ) - \nabla _ { \mathbf { x } } \log q _ { t } ^ { \prime } ( \mathbf { x } ) \| _ { 2 } ^ { 2 } + C _ { 1 } } + \sqrt { 2 - 2 \exp \{ - C _ { 2 } \| \delta \| _ { 2 } ^ { 2 } \} } + \epsilon _ { r e } } } \\ { { \displaystyle C _ { 1 } = ( L _ { u } + 8 M ^ { 2 } ) \int _ { t } ^ { T } \beta ( t ) d t , C _ { 2 } = ( 8 ( 1 - \Pi _ { s = 1 } ^ { t } ( 1 - \beta _ { s } ) ) ) ^ { - 1 } . } } \end{array}
$$

Proof sketch. We first use the triangular inequality to upper bound $D _ { T V } \big ( q _ { t } , q _ { t } ^ { \prime } \big )$ with $D _ { T V } ( q _ { t } , p _ { t } ) +$ $D _ { T V } ( p _ { t } , p _ { t } ^ { \prime } ) + D _ { T V } ( p _ { t } ^ { \prime } , q _ { t } ^ { \prime } )$ . $D _ { T V } ( q _ { t } , p _ { t } )$ can be upper bounded by a function of the Hellinger distance $H ( q _ { t } , p _ { t } )$ , which can be calculated explicitly. $D _ { T V } ( p _ { t } , p _ { t } ^ { \prime } )$ can be upper bounded by the reconstruction error $\epsilon _ { r e }$ by assumption. To upper bound $D _ { T V } ( p _ { t } ^ { \prime } , q _ { t } ^ { \prime } )$ , we can leverage Pinker’s inequality to alternatively upper bound the KL-divergence between $p _ { t } ^ { \prime }$ and $q _ { t } ^ { \prime }$ which can be derived by using the Fokker-Planck equation [44] in the reverse SDE.

Remark. A large deviated-reconstruction loss can indicate a large total variation distance $D _ { T V } ( q _ { t } , q _ { t } ^ { \prime } )$ , which is the lower bound of a function with respect to the score-matching loss $\| \mathbf { E } _ { t , \mathbf { x } } \| s _ { \theta } ( \mathbf { x } , t ) -$ $\nabla _ { \mathbf { x } } \log q _ { t } ^ { \prime } ( \mathbf { x } ) \lVert _ { 2 } ^ { 2 }$ (in RHS of Equation (9)). Therefore, we show that maximizing the deviatedreconstruction loss implicitly maximizes the score-matching loss, and thus induces inaccurate data density estimation to perform an effective attack. The connection of deviated-reconstruction loss and the density gradient loss for DDPM is provided in Thm. 3 in Appendix C.2.

# 3.3 Segment-wise forwarding-backwarding algorithm

Adaptive attacks against diffusion-based purification require gradient backpropagation through the forwarding path. For diffusion-based purification, the memory cost scales linearly with the diffusion length $T$ and is not feasible in a realistic application. Therefore, existing defenses either use a surrogate model for gradient approximation [55, 56, 60, 45] or consider adaptive attacks only for a small diffusion length [4], but the approximation can induce error and downgrade the attack performance a lot. Recently, DiffPure [34] leverages the adjoint method [28] to backpropagate the gradient of SDE within reasonable memory cost and enables adaptive attacks against score-based purification. However, it cannot be applied to a discrete process, and the memory-efficient gradient backpropagation algorithm is unexplored for DDPM. Another line of research [9, 8, 19] proposes the technique of gradient checkpointing to perform gradient backpropagation with memory efficiency. Fewer activations are stored during forwarding passes, and the local computation graph is constructed via recomputation. However, we are the first to apply the memory-efficient backpropagation technique to attack diffusion purification defenses and resolve the problem of memory cost during attacks, which is realized as a challenging problem by prior attacks against purification defenses [34, 60]. Concretely, we propose a segment-wise forwarding-backwarding algorithm, which leads to memory-efficient gradient computation of the attack loss with respect to the adversarial examples.

We first feed the input $\mathbf { x } _ { \mathrm { 0 } }$ to the diffusion-based purification process and store the intermediate samples $\mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , . . . , \mathbf { x } _ { T }$ in the diffusion process and $\mathbf { x } _ { T } ^ { \prime } , \mathbf { \bar { x } } _ { T - 1 } ^ { \prime } , . . . , \mathbf { \bar { x } } _ { 0 } ^ { \prime }$ in the reverse process sequentially. For ease of notation, we have $\mathbf { x } _ { t + 1 } = f _ { d } ( \mathbf { x } _ { t } )$ and $\mathbf { x } _ { t } ^ { \prime } = f _ { r } ( \mathbf { x } _ { t + 1 } ^ { \prime } )$ for $t \in [ 0 , T - 1 ]$ . Then we can backpropagate the gradient iteratively following:

$$
\frac { \partial \mathcal { L } } { \partial \mathbf { x } _ { t + 1 } ^ { \prime } } = \frac { \partial \mathcal { L } } { \partial \mathbf { x } _ { t } ^ { \prime } } \frac { \partial \mathbf { x } _ { t } ^ { \prime } } { \partial \mathbf { x } _ { t + 1 } ^ { \prime } } = \frac { \partial \mathcal { L } } { \partial \mathbf { x } _ { t } ^ { \prime } } \frac { \partial f _ { r } ( \mathbf { x } _ { t + 1 } ^ { \prime } ) } { \partial \mathbf { x } _ { t + 1 } ^ { \prime } }
$$

At each time step $t$ in the reverse process, we only need to store the gradient $\partial \mathcal { L } / \partial \mathbf { x } _ { t } ^ { \prime }$ , the intermediate sample $\mathbf { x } _ { t + 1 } ^ { \prime }$ and the model $f _ { r }$ to construct the computational graph. When we backpropagate the gradients at the next time step $t + 1$ , the computational graph at time step $t$ will no longer be reused, and thus, we can release the memory of the graph at time step $t$ . Therefore, we only have one segment of the computational graph used for gradient backpropagation in the memory at each time step. We can similarly backpropagate the gradients in the diffusion process. Ignoring the memory cost of storing intermediate samples (usually small compared to the memory cost of computational graphs), the memory cost of our segment-wise forwarding-backwarding algorithm is $\mathcal { O } ( 1 )$ (validated in Figure 3).

We summarize the detailed procedures in Algorithm 1 in Appendix B. It can be applied to gradient backpropagation through any discrete Markov process with a long path. Basically, we $^ { l }$ ) perform the forward pass and store the intermediate samples, 2) allocate the memory of one segment of the computational graph in the memory and simulate the forwarding pass of the segment with intermediate samples, 3) backpropagate the gradients through the segment and release the memory of the segment, and 4) go to step 2 and consider the next segment until termination.

# 3.4 DiffAttack Technique

Currently, AutoAttack [12] holds the state-of-the-art attack algorithm, but it fails to attack the diffusion-based purification defenses due to the challenge of vanishing/exploding gradient, memory cost and large randomness. To specifically tackle the challenges, we integrate the deviatedreconstruction loss (in Section 3.2) and the segment-wise forwarding-backwarding algorithm (in Section 3.3) as an attack technique DiffAttack against diffusion-based purification, including the scorebased and DDPM-based purification defenses. The pictorial illustration of DiffAttack is provided in Figure 1.

Concretely, we maximize the surrogate loss $\mathcal { L }$ as the optimization objective in Equation (7):

$$
\operatorname* { m a x } \mathcal { L } = \mathcal { L } _ { c l s } + \lambda \mathcal { L } _ { d e v }
$$

where $\mathcal { L } _ { c l s }$ is the CE loss or DLR loss, $\mathcal { L } _ { d e v }$ is the deviated-reconstruction loss formulated in Equation (8), and $\lambda$ is the weight coefficient. During the optimization, we use the segment-wise forwarding-backwarding algorithm for memory-efficient gradient backpropagation. Note that $\mathcal { L } _ { d e v }$ suffers less from the gradient problem compared with $\mathcal { L } _ { c l s }$ , and thus the objective of $\mathcal { L } _ { d e v }$ can be optimized more precisely and stably, but it does not resolve the gradient problem of $\mathcal { L } _ { c l s }$ . On the other hand, the optimization of $\mathcal { L } _ { d e v }$ benefits the optimization of $\mathcal { L } _ { c l s }$ in the sense that $\mathcal { L } _ { d e v }$ can induce a deviated reconstruction of the image with a larger probability of misclassification. $\lambda$ controls the balance of the two objectives. A small $\lambda$ can weaken the deviated-reconstruction object and make the attack suffer more from the vanishing/exploded gradient problem, while a large $\lambda$ can downplay the guidance of the classification loss and confuse the direction towards the decision boundary of the classifier.

Attack against randomized diffusion-based purification. DiffAttack tackles the randomness problem from two perspectives: 1) sampling the diffused and reconstructed samples across different time steps multiple times as in Equation (8) (similar to EOT [3]), and 2) optimizing perturbations for all samples including misclassified ones in all steps. Perspective 1) provides a more accurate estimation of gradients against sample variance of the diffusion process. Perspective 2) ensures a more effective and stable attack optimization since the correctness of classification is of high variance over different steps in the diffusion purification setting. Formally, the classification result of a sample can be viewed as a Bernoulli distribution (i.e., correct or false). We should reduce the success rate of the Bernoulli distribution of sample classification by optimizing them with a larger attack loss, which would lead to lower robust accuracy. In other words, one observation of failure in classification does not indicate that the sample has a low success rate statistically, and thus, perspective 2) helps to continue optimizing the perturbations towards a lower success rate (i.e., away from the decision boundary). We provide the pseudo-codes of DiffAttack in Algorithm 2 in Appendix D.1.

Table 1: Attack performance (Rob-Acc $( \% )$ ) of DiffAttack and AdjAttack [34] against score-based purification on CIFAR-10.   

<table><tr><td>Models</td><td>T</td><td>Cl-Acc</td><td>ep Attack</td><td>E</td><td>Method</td><td>Rob-Acc</td><td>Diff.</td></tr><tr><td rowspan="3">WideResNet-28-10</td><td rowspan="2">0.1</td><td rowspan="2">89.02</td><td rowspan="2">l8</td><td>8/255</td><td>AdjAttack DiffAttack</td><td>70.64 46.88</td><td>-23.76</td></tr><tr><td>4/255</td><td>AdjAttack DiffAttack</td><td>82.81 71.88</td><td>-10.93</td></tr><tr><td>0.075</td><td>91.03</td><td>l2</td><td>0.5</td><td>AdjAttack DiffAttack</td><td>78.58 64.06</td><td>-14.52</td></tr><tr><td rowspan="3">WideResNet-70-16</td><td>0.1</td><td>90.07</td><td>l8</td><td>8/255</td><td>AdjAttack DiffAttack AdjAttack</td><td>71.29 45.31 81.25</td><td>-25.98</td></tr><tr><td>0.075</td><td>92.68</td><td>l2</td><td>4/255 0.5</td><td>DiffAttack AdjAttack</td><td>75.00 80.60</td><td>-6.25</td></tr><tr><td></td><td></td><td></td><td></td><td>DiffAttack</td><td>70.31</td><td>-10.29</td></tr></table>

# 4 Experimental Results

In this section, we evaluate DiffAttack from various perspectives empirically. As a summary, we find that 1) DiffAttack significantly outperforms other SOTA attack methods against diffusion-based defenses on both the score-based purification and DDPM-based purification models, especially under large perturbation radii (Section 4.2 and Section 4.3); 2) DiffAttack outperforms other strong attack methods such as the black-box attack and adaptive attacks against other adversarial purification defenses (Section 4.4); 3) a moderate diffusion length $T$ benefits the model robustness, since too long/short diffusion length would hurt the robustness (Section 4.5); 4) our proposed segment-wise forwarding-backwarding algorithm achieves $\mathcal { O } ( 1 )$ -memory cost and outperforms other baselines by a large margin (Section 4.6); and 5) attacks with the deviated-reconstruction loss added over uniformly sampled time steps outperform that added over only initial/final time steps (Section 4.7).

# 4.1 Experiment Setting

Dataset & model. We validate DiffAttack on CIFAR-10 [27] and ImageNet [13]. We consider different network architectures for classification. Particularly, WideResNet-28-10 and WideResNet-70-16 [61] are used on CIFAR-10, and ResNet-50 [23], WideResNet-50-2 (WRN-50-2), and ViT (DeiT-S) [16] are used on ImageNet. We use a pretrained score-based diffusion model [49] and DDPM [25] to purify images following [34, 4].

Evaluation metric. The performance of attacks is evaluated using the robust accuracy (Rob-Acc), which measures the ratio of correctly classified instances over the total number of test data under certain perturbation constraints. Following the literature [12], we consider both $\ell _ { \infty }$ and $\ell _ { 2 }$ attacks under multiple perturbation constraints $\epsilon$ . We also report the clean accuracy (Cl-Acc) for different approaches.

Baselines. To demonstrate the effectiveness of DiffAttack, we compare it with 1) SOTA attacks against score-based diffusion adjoint attack (AdjAttack) [34], 2) SOTA attack against DDPM-based diffusion Diff-BPDA attack [4], 3) SOTA black-box attack SPSA [53] and square attack [1], and 4) specific attack against EBM-based purification joint attack [60]. We defer more explanations of baselines and experiment details to Appendix D.2. The codes are publicly available at https: //github.com/kangmintong/DiffAttack.

Table 3: Attack performance (Rob-Acc $( \% )$ ) of DiffAttack and Diff-BPDA [4] against DDPM-based purification on CIFAR-10.   

<table><tr><td>Architecture</td><td>T</td><td>C1-Acc</td><td>lp Attack</td><td>E</td><td>Method</td><td>Rob-Acc</td><td>Diff.</td></tr><tr><td rowspan="4">WideResNet-28-10</td><td rowspan="2">100</td><td rowspan="2">87.50</td><td rowspan="2">l8</td><td rowspan="2">8/255</td><td>Diff-BPDA</td><td>75.00</td><td rowspan="2">-20.31</td></tr><tr><td>DiffAttack</td><td>54.69</td></tr><tr><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">4/255</td><td>Diff-BPDA DiffAttack</td><td>76.56 63.28</td><td rowspan="2">-13.28</td></tr><tr><td>Diff-BPDA</td><td>76.56</td></tr><tr><td rowspan="5"></td><td rowspan="5"></td><td rowspan="5">90.62</td><td rowspan="5">l2 lo</td><td rowspan="2">0.5</td><td>DiffAttack</td><td>67.97</td><td rowspan="2">-8.59</td></tr><tr><td>Diff-BPDA</td><td>74.22</td></tr><tr><td rowspan="2">8/255</td><td>DiffAttack</td><td>59.38</td><td rowspan="2">-14.84</td></tr><tr><td>Diff-BPDA</td><td>75.78</td></tr><tr><td rowspan="2"></td><td rowspan="2">4/255 0.5</td><td>DiffAttack Diff-BPDA</td><td>67.19 81.25</td><td rowspan="2">-8.59 -9.37</td></tr><tr><td>92.19</td><td>DiffAttack</td><td>71.88</td></tr></table>

# 4.2 Attack against score-based purification

DiffPure [34] presents the state-of-theart adversarial purification performance using the score-based diffusion models [49]. It proposes a strong adaptive attack (AdjAttack) which uses the adjoint method [28] to efficiently backpropagate the gradients through reverse SDE. Therefore, we select AdjAttack as the strong baseline and compare DiffAttack with it. The results on CIFAR-10 in Table 1 show that DiffAttack achieves much lower robust accuracy compared with AdjAttack under different types of attacks ( $\ell _ { \infty }$ and $\ell _ { 2 }$ attack) with multiple perturbation constraints $\epsilon$ . Concretely, DiffAttack decreases the robust accuracy

Table 2: Attack performance of DiffAttack and AdjAttack [34] against score-based adversarial purification with diffusion length $T = 0 . 0 1 5$ on ImageNet under $\ell _ { \infty }$ attack $\zeta = 4 / 2 5 \bar { 5 } )$ .   

<table><tr><td>Models</td><td>C1-Acc</td><td>Method</td><td>Rob-Acc</td><td>Diff.</td></tr><tr><td>ResNet-50</td><td>67.79</td><td>AdjAttack DiffAttack</td><td>40.93 28.13</td><td>-12.80</td></tr><tr><td>WRN-50-2</td><td>71.16</td><td>AdjAttack DiffAttack</td><td>44.39 31.25</td><td>-13.14</td></tr><tr><td>DeiT-S</td><td>73.63</td><td>AdjAttack DiffAttack</td><td>43.18 32.81</td><td>-10.37</td></tr></table>

of models by over $2 0 \%$ under $\ell _ { \infty }$ attack with $\epsilon = 8 / 2 5 5$ $( 7 0 . 6 4 \% \to 4 6 . 8 8 \%$ on WideResNet-28-10 and $7 1 . 2 9 \% \to 4 5 . 3 1 \%$ on WideResNet-70-16). The effectiveness of DiffAttack also generalizes well to large-scale datasets ImageNet as shown in Table 2. Note that the robust accuracy of the state-of-the-art non-diffusion-based purification defenses [38, 21] achieve about $6 5 \%$ robust accuracy on CIFAR-10 with WideResNet-28-10 under $\ell _ { \infty } = 8 / 2 5 5$ attack $\zeta = 8 / 2 5 5 )$ , while the performance of score-based purification under AdjAttack in the same setting is $7 0 . 6 4 \%$ . However, given the strong DiffAttack, the robust accuracy of score-based purification drops to $4 6 . 8 8 \%$ . It motivates us to think of more effective techniques to further improve the robustness of diffusion-based purification in future work.

# 4.3 Attack against DDPM-based purification

Another line of diffusion-based purification defenses [4, 55, 56] leverages DDPM [46] to purify the images with intentionally crafted perturbations. Since backpropagating the gradients along the diffusion and sampling process with a relatively large diffusion length is unrealistic due to the large memory cost, BPDA attack [2] is adopted as the strong attack against the DDPM-based purification. However, with our proposed segment-wise forwarding-backwarding algorithm, we can compute the gradients within a small budget of memory cost, and to our best knowledge, this is the first work to achieve adaptive gradient-based adversarial attacks against DDPM-based purification. We compare DiffAttack with Diff-BPDA attack [4] on CIFAR-10, and the results in Table 3 demonstrate that DiffAttack outperforms the baseline by a large margin under both $\ell _ { \infty }$ and $\ell _ { 2 }$ attacks.

# 4.4 Comparison with other adaptive attack methods

Besides the AdjAttack and Diff-BPDA attacks against existing diffusion-based purification defenses, we also compare DiffAttack with other general types of adaptive attacks: 1) black-box attack SPSA [53] and 2) square attack [1], as well as 3) adaptive attack against score-based generative models joint attack (Score / Full) [60]. SPSA attack approximates the gradients by randomly sampling from a pre-defined distribution and using the finite-difference method. Square attack heuristically searches for adversarial examples in a

Table 4: Robust accuracy $( \% )$ of DiffAttack compared with other attack methods on CIFAR-10 with WideResNet-28-10 under $\ell _ { \infty }$ attack $\acute { \epsilon } = 8 / 2 5 5$ ).   

<table><tr><td>Method</td><td>Score-based</td><td>DDPM-based</td></tr><tr><td>SPSA</td><td>83.37</td><td>81.29</td></tr><tr><td>Square Attack</td><td>82.81</td><td>81.68</td></tr><tr><td>Joint Attack (Score)</td><td>72.74</td><td>1</td></tr><tr><td>Joint Attack (Full)</td><td>77.83</td><td>76.26</td></tr><tr><td>Diff-BPDA</td><td>78.13</td><td>75.00</td></tr><tr><td>AdjAttack</td><td>70.64</td><td>1</td></tr><tr><td>DiffAttack</td><td>46.88</td><td>54.69</td></tr></table>

low-dimensional space with the constraints of perturbation patterns. Joint attack (score) updates the input by the average of the classifier gradient and the output of the score estimation network, while joint attack (full) leverages the classifier gradients and the difference between the input and the purified samples. The results in Table 4 show that DiffAttack outperforms SPSA, square attack, and joint attack by a large margin on score-based and DDPM-based purification defenses. Note that joint attack (score) cannot be applied to the DDPM-based pipeline due to the lack of a score estimator. AdjAttack fails on the DDPM-based pipeline since it can only calculate gradients through SDE.

# 4.5 Robustness with different diffusion lengths

We observe that the diffusion length plays an extremely important role in the effectiveness of adversarial purification. Existing DDPM-based purification works [56, 55] prefer a small diffusion length, but we find it vulnerable under our DiffAttack. The influence of the diffusion length $T$ on the performance (clean/robust accuracy) of the purification defense methods is illustrated in Figure 2. We observe that $I$ ) the clean accuracy of the purification

![](images/2009deede73d78dafd94869cc35a1591d0ddd8e06871d9d427ca5ca06f6623f0.jpg)  
Figure 2: The clean/robust accuracy $( \% )$ of diffusion-based purification with different diffusion length $T$ under DiffAttack on CIFAR-10 with WideResNet-28-10 under $\ell _ { \infty }$ attack $\epsilon = 8 / 2 5 5$ .

defenses negatively correlates with the diffusion lengths since the longer diffusion process adds more noise to the input and induces inaccurate reconstruction of the input sample; and 2) a moderate diffusion length benefits the robust accuracy since diffusion-based purification with a small length makes it easier to compute the gradients for attacks, while models with a large diffusion length have poor clean accuracy that deteriorates the robust accuracy. We also validate the conclusion on ImageNet in Appendix D.3.

# 4.6 Comparison of memory cost

Recent work [4] computes the gradients of the diffusion and sampling process to perform the gradient-based attack, but it only considers a small diffusion length (e.g., 14 on CIFAR-10). They construct the computational graph once and for all, which is extremely expensive for memory cost with a large diffusion length. We use a segment-wise forwarding-backwarding algorithm in Section 3.3 to avoid allocating the memory for the whole computational graph. In this part, we validate the memory efficiency of our approach compared to [4]. The results

![](images/fb5d12de6e7e8a2c8378daa73ef836e6f843dd6cbe53861ff26a66a0c0e16ae7.jpg)  
Figure 3: Comparison of memory cost of gradient backpropagation between [4] and DiffAttack with batch size 16 on CIFAR-10 with WideResNet-28-10 under $\ell _ { \infty }$ attack.

in Figure 3 demonstrate that 1) the gradient backpropagation of [4] has the memory cost linearly correlated to the diffusion length and does not scale up to the diffusion length of 30, while 2) DiffAttack has almost constant memory cost and is able to scale up to extremely large diffusion length $\begin{array} { r } { T = 1 0 0 0 \mathrm { \Omega } } { } \end{array}$ ). The evaluation is done on an RTX A6000 GPU. In Appendix D.3, we provide comparisons of runtime between DiffAttack and [4] and demonstrate that DiffAttack reduces the memory cost with comparable runtime.

# 4.7 Influence of applying the deviated-reconstruction loss at different time steps

We also show that the time steps at which we apply the deviated-reconstruction loss also influence the effectiveness of DiffAttack. Intuitively, the loss added at small time steps does not suffer from vanishing/exploding gradients but lacks supervision at consequent time steps, while the loss added at large time steps gains strong supervision but suffers from the gradient problem. The results in Figure 4 show that adding deviated-reconstruction loss to uniformly sampled time steps $( \mathrm { U n i } ( 0 , \mathrm { T } ) )$ achieves the best attack performance and tradeoff compared with that of adding loss to the same number of partial time steps only at the initial stage $( ( 0 , T / 3 ) ,$ ) or the final stage $( ( 2 T / 3 , T ) )$ . For fair comparisons, we uniformly sample $T / 3$ time steps (identical to partial stage guidance $( 0 , T / 3 )$ , $( 2 T / 3 , T ) )$ to impose ${ \mathcal { L } } _ { \mathrm { d e v } }$ .

![](images/3b86efeca9f0e30262a09948122d4b1567a41fbed339ca8bac4418eef9804d24.jpg)  
Figure 4: The impact of applying $\mathcal { L } _ { d e v }$ at different time steps on decreased robust accuracy $( \% )$ . $T$ is the diffusion length and $\mathrm { U n i } ( 0 , T )$ represents uniform sampling.

# 5 Related Work

Adversarial purification methods purify the adversarial input before classification with generative models. Defense-gan [43] trains a GAN to restore the clean samples. Pixeldefend [48] utilizes an autoregressive model to purify adversarial examples. Another line of research [50, 22, 17, 24, 60] leverages energy-based model (EBM) and Markov chain Monte Carlo (MCMC) to perform the purification. More recently, diffusion models have seen wide success in image generation [15, 40, 41, 42, 31, 39]. They are also used to adversarial purification [34, 4, 62, 57, 51, 55, 56] and demonstrated to achieve the state-of-the-art robustness. In this work, we propose DiffAttack specifically against diffusion-based purification and show the effectiveness in different settings, which motivates future work to improve the robustness of the pipeline.

Adversarial attacks search for visually imperceptible signals which can significantly perturb the prediction of models [52, 20]. Different kinds of defense methods are progressively broken by advanced attack techniques, including white-box attack [6, 2, 32] and black-box attack [1, 53, 35]. [11, 12, 37, 59, 29] propose a systematic and automatic framework to attack existing defense methods. Despite attacking most defense methods, these approaches are shown to be ineffective against the diffusion-based purification pipeline due to the problem of vanishing/exploding gradient, memory cost, and randomness. Therefore, we propose DiffAttack to specifically tackle the challenges and successfully attack the diffusion-based purification defenses.

# 6 Conclusion

In this paper, we propose DiffAttack, including the deviated-reconstruction loss added on intermediate samples and a segment-wise forwarding-backwarding algorithm. We empirically demonstrate that DiffAttack outperforms existing adaptive attacks against diffusion-based purification by a large margin. We conduct a series of ablation studies and show that a moderate diffusion length benefits the model robustness, and attacks with the deviated-reconstruction loss added over uniformly sampled time steps outperform that added over only initial/final time steps, which will help to better understand the properties of diffusion process and motivate future defenses.

Acknolwdgement. This work is partially supported by the National Science Foundation under grant No. 1910100, No. 2046726, No. 2229876, DARPA GARD, the National Aeronautics and Space Administration (NASA) under grant no. 80NSSC20M0229, the Alfred P. Sloan Fellowship, the Amazon research award, and the eBay research award.

References   
[1] Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion, and Matthias Hein. Square attack: a query-efficient black-box adversarial attack via random search. In European Conference on Computer Vision, pages 484–501. Springer, 2020.   
[2] Anish Athalye, Nicholas Carlini, and David Wagner. Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. In International conference on machine learning, pages 274–283. PMLR, 2018.   
[3] Anish Athalye, Logan Engstrom, Andrew Ilyas, and Kevin Kwok. Synthesizing robust adversarial examples. In International conference on machine learning, pages 284–293. PMLR, 2018.   
[4] Tsachi Blau, Roy Ganz, Bahjat Kawar, Alex Bronstein, and Michael Elad. Threat modelagnostic adversarial defense using diffusion models. arXiv preprint arXiv:2207.08089, 2022.   
[5] Yulong Cao, Ningfei Wang, Chaowei Xiao, Dawei Yang, Jin Fang, Ruigang Yang, Qi Alfred Chen, Mingyan Liu, and Bo Li. Invisible for both camera and lidar: Security of multi-sensor fusion based perception in autonomous driving under physical-world attacks. In 2021 IEEE Symposium on Security and Privacy (SP), pages 176–194. IEEE, 2021.   
[6] Nicholas Carlini and David Wagner. Adversarial examples are not easily detected: Bypassing ten detection methods. In Proceedings of the 10th ACM workshop on artificial intelligence and security, pages 3–14, 2017.   
[7] Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In 2017 ieee symposium on security and privacy (sp), pages 39–57. Ieee, 2017.   
[8] Bo Chang, Lili Meng, Eldad Haber, Lars Ruthotto, David Begert, and Elliot Holtham. Reversible architectures for arbitrarily deep residual neural networks. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.   
[9] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174, 2016.   
[10] Zhaoyu Chen, Bo Li, Shuang Wu, Kaixun Jiang, Shouhong Ding, and Wenqiang Zhang. Content-based unrestricted adversarial attack. arXiv preprint arXiv:2305.10665, 2023.   
[11] Francesco Croce, Maksym Andriushchenko, Vikash Sehwag, Edoardo Debenedetti, Nicolas Flammarion, Mung Chiang, Prateek Mittal, and Matthias Hein. Robustbench: a standardized adversarial robustness benchmark. arXiv preprint arXiv:2010.09670, 2020.   
[12] Francesco Croce and Matthias Hein. Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. In International conference on machine learning, pages 2206–2216. PMLR, 2020.   
[13] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.   
[14] Luc Devroye, Abbas Mehrabian, and Tommy Reddad. The total variation distance between high-dimensional gaussians. arXiv preprint arXiv:1810.08693, 2018.   
[15] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in Neural Information Processing Systems, 34:8780–8794, 2021.   
[16] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.   
[17] Yilun Du and Igor Mordatch. Implicit generation and modeling with energy based models. Advances in Neural Information Processing Systems, 32, 2019.   
[18] Kevin Eykholt, Ivan Evtimov, Earlence Fernandes, Bo Li, Amir Rahmati, Chaowei Xiao, Atul Prakash, Tadayoshi Kohno, and Dawn Song. Robust physical-world attacks on deep learning visual classification. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1625–1634, 2018.   
[19] Aidan N Gomez, Mengye Ren, Raquel Urtasun, and Roger B Grosse. The reversible residual network: Backpropagation without storing activations. Advances in neural information processing systems, 30, 2017.   
[20] Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.   
[21] Sven Gowal, Sylvestre-Alvise Rebuffi, Olivia Wiles, Florian Stimberg, Dan Andrei Calian, and Timothy A Mann. Improving robustness using generated data. Advances in Neural Information Processing Systems, 34:4218–4233, 2021.   
[22] Will Grathwohl, Kuan-Chieh Wang, Jörn-Henrik Jacobsen, David Duvenaud, Mohammad Norouzi, and Kevin Swersky. Your classifier is secretly an energy based model and you should treat it like one. arXiv preprint arXiv:1912.03263, 2019.   
[23] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.   
[24] Mitch Hill, Jonathan Mitchell, and Song-Chun Zhu. Stochastic security: Adversarial defense using long-run dynamics of energy-based models. arXiv preprint arXiv:2005.13525, 2020.   
[25] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020.   
[26] Aapo Hyvärinen and Peter Dayan. Estimation of non-normalized statistical models by score matching. Journal of Machine Learning Research, 6(4), 2005.   
[27] Alex Krizhevsky. Learning multiple layers of features from tiny images. 2009.   
[28] Xuechen Li, Ting-Kam Leonard Wong, Ricky TQ Chen, and David Duvenaud. Scalable gradients for stochastic differential equations. In International Conference on Artificial Intelligence and Statistics, pages 3870–3882. PMLR, 2020.   
[29] Xiang Ling, Shouling Ji, Jiaxu Zou, Jiannan Wang, Chunming Wu, Bo Li, and Ting Wang. Deepsec: A uniform platform for security analysis of deep learning model. In 2019 IEEE Symposium on Security and Privacy (SP), pages 673–690. IEEE, 2019.   
[30] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations, 2018.   
[31] Chenlin Meng, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021.   
[32] Marius Mosbach, Maksym Andriushchenko, Thomas Trost, Matthias Hein, and Dietrich Klakow. Logit pairing methods can fool gradient-based attacks. arXiv preprint arXiv:1810.12042, 2018.   
[33] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International Conference on Machine Learning, pages 8162–8171. PMLR, 2021.   
[34] Weili Nie, Brandon Guo, Yujia Huang, Chaowei Xiao, Arash Vahdat, and Anima Anandkumar. Diffusion models for adversarial purification. In International Conference on Machine Learning (ICML), 2022.   
[35] Nicolas Papernot, Patrick McDaniel, Ian Goodfellow, Somesh Jha, Z Berkay Celik, and Ananthram Swami. Practical black-box attacks against machine learning. In Proceedings of the 2017 ACM on Asia conference on computer and communications security, pages 506–519, 2017.   
[36] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019.   
[37] Maura Pintor, Luca Demetrio, Angelo Sotgiu, Ambra Demontis, Nicholas Carlini, Battista Biggio, and Fabio Roli. Indicators of attack failure: Debugging and improving optimization of adversarial examples. arXiv preprint arXiv:2106.09947, 2021.   
[38] Sylvestre-Alvise Rebuffi, Sven Gowal, Dan A Calian, Florian Stimberg, Olivia Wiles, and Timothy Mann. Fixing data augmentation to improve adversarial robustness. arXiv preprint arXiv:2103.01946, 2021.   
[39] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.   
[40] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad Norouzi. Palette: Image-to-image diffusion models. In ACM SIGGRAPH 2022 Conference Proceedings, pages 1–10, 2022.   
[41] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models with deep language understanding. arXiv preprint arXiv:2205.11487, 2022.   
[42] Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi. Image super-resolution via iterative refinement. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.   
[43] Pouya Samangouei, Maya Kabkab, and Rama Chellappa. Defense-gan: Protecting classifiers against adversarial attacks using generative models. arXiv preprint arXiv:1805.06605, 2018.   
[44] Simo Särkkä and Arno Solin. Applied stochastic differential equations, volume 10. Cambridge University Press, 2019.   
[45] Changhao Shi, Chester Holtz, and Gal Mishne. Online adversarial purification based on self-supervision. arXiv preprint arXiv:2101.09387, 2021.   
[46] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, pages 2256–2265. PMLR, 2015.   
[47] Yang Song, Sahaj Garg, Jiaxin Shi, and Stefano Ermon. Sliced score matching: A scalable approach to density and score estimation. In Uncertainty in Artificial Intelligence, pages 574–584. PMLR, 2020.   
[48] Yang Song, Taesup Kim, Sebastian Nowozin, Stefano Ermon, and Nate Kushman. Pixeldefend: Leveraging generative models to understand and defend against adversarial examples. arXiv preprint arXiv:1710.10766, 2017.   
[49] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021.   
[50] Vignesh Srinivasan, Csaba Rohrer, Arturo Marban, Klaus-Robert Müller, Wojciech Samek, and Shinichi Nakajima. Robustifying models against adversarial attacks by langevin dynamics. Neural Networks, 137:1–17, 2021.   
[51] Jiachen Sun, Weili Nie, Zhiding Yu, Z Morley Mao, and Chaowei Xiao. Pointdp: Diffusiondriven purification against adversarial attacks on 3d point cloud recognition. arXiv preprint arXiv:2208.09801, 2022.   
[52] Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199, 2013.   
[53] Jonathan Uesato, Brendan O’Donoghue, Pushmeet Kohli, and Aäron van den Oord. Adversarial risk and the dangers of evaluating against weak attacks. In icml2018, 2018.   
[54] Boxin Wang, Weixin Chen, Hengzhi Pei, Chulin Xie, Mintong Kang, Chenhui Zhang, Chejian Xu, Zidi Xiong, Ritik Dutta, Rylan Schaeffer, et al. Decodingtrust: A comprehensive assessment of trustworthiness in gpt models. arXiv preprint arXiv:2306.11698, 2023.   
[55] Jinyi Wang, Zhaoyang Lyu, Dahua Lin, Bo Dai, and Hongfei Fu. Guided diffusion model for adversarial purification. arXiv preprint arXiv:2205.14969, 2022.   
[56] Quanlin Wu, Hang Ye, and Yuntian Gu. Guided diffusion model for adversarial purification from random noise. arXiv preprint arXiv:2206.10875, 2022.   
[57] Chaowei Xiao, Zhongzhu Chen, Kun Jin, Jiongxiao Wang, Weili Nie, Mingyan Liu, Anima Anandkumar, Bo Li, and Dawn Song. Densepure: Understanding diffusion models towards adversarial robustness. arXiv preprint arXiv:2211.00322, 2022.   
[58] Haotian Xue, Alexandre Araujo, Bin Hu, and Yongxin Chen. Diffusion-based adversarial sample generation for improved stealthiness and controllability. arXiv preprint arXiv:2305.16494, 2023.   
[59] Chengyuan Yao, Pavol Bielik, Petar Tsankov, and Martin Vechev. Automated discovery of adaptive attacks on adversarial defenses. Advances in Neural Information Processing Systems, 34:26858–26870, 2021.   
[60] Jongmin Yoon, Sung Ju Hwang, and Juho Lee. Adversarial purification with score-based generative models. In International Conference on Machine Learning, pages 12062–12072. PMLR, 2021.   
[61] Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. arXiv preprint arXiv:1605.07146, 2016.   
[62] Kui Zhang, Hang Zhou, Jie Zhang, Qidong Huang, Weiming Zhang, and Nenghai Yu. Ada3diff: Defending against 3d adversarial point clouds via adaptive diffusion. arXiv preprint arXiv:2211.16247, 2022.

# A Broader Impact and Limitations

Broader impact. As an effective and popular way to explore the vulnerabilities of ML models, adversarial attacks have been widely studied. However, recent diffusion-based purification is shown hard to attack based on different trials, which raises an interesting question of whether it can be attacked. Our paper provides the first effective attack against such defenses to identify the vulnerability of diffusion-based purification for the community and inspire more effective defense approaches. In particular, we propose an effective evasion attack against diffusion-based purification defenses which consists of a deviated-reconstruction loss at intermediate diffusion steps to induce inaccurate density gradient estimation and a segment-wise forwarding-backwarding algorithm to achieve memoryefficient gradient backpropagation. The effectiveness of the deviated-reconstruction loss helps us to better understand the properties of diffusion purification. Concretely, there exist adversarial regions in the intermediate sample space where the score approximation model outputs inaccurate density gradients and finally misleads the prediction. The observation motivates us to design a more robust sampling process in the future, and one potential way is to train a more robust score-based model. Furthermore, the segment-wise forwarding-backwarding algorithm tackles the memory issue of gradient propagation through a long path. It can be applied to the gradient calculation of any discrete Markov process almost within a constant memory cost. To conclude, our attack motivates us to rethink the robustness of a line of SOTA diffusion-based purification defenses and inspire more effective defenses.

Limitations. In this paper, we propose an effective attack algorithm DiffAttack against diffusionbased purification defenses. A possible negative societal impact may be the usage of DiffAttack in safety-critical scenarios such as autonomous driving and medical imaging analysis to mislead the prediction of machine learning models. However, the foundation of DiffAttack and important findings about the diffusion process properties can benefit our understanding of the vulnerabilities of diffusionbased purification defenses and therefore motivate more effective defenses in the future. Concretely, the effectiveness of DiffAttack indicates that there exist adversarial regions in the intermediate sample space where the score approximation model outputs inaccurate density gradients and finally misleads the prediction. The observation motivates us to design a more robust sampling process in the future, and one potential way is to train a more robust score-based model. Furthermore, to control a robust sampling process, it is better to provide guidance across uniformly sampled time steps rather than only at the final stage according to our findings.

# Algorithm 1 Segment-wise forwarding-backwarding algorithm (PyTorch-like pseudo-codes)

1: Input: $f _ { r }$ , fd, ∂L/∂x′0, xi, x′i (i ∈ [T ])   
2: Output: $\partial \mathcal { L } / \partial \mathbf { x } _ { 0 }$   
3: for $t = 1$ to $T$ do   
4: Creat_Graph $( f _ { r } ( \mathbf { x } _ { t } ^ { \prime } )  \mathbf { x } _ { t - 1 } ^ { \prime } )$ )   
5: $\mathcal { L } ^ { \prime } \gets \left( \partial \mathcal { L } / \partial \mathbf { x } _ { t - 1 } ^ { \prime } \right) \left( f _ { r } ( \mathbf { x } _ { t } ^ { \prime } ) \right)$   
6: $\partial \mathcal { L } / \partial \mathbf { x } _ { t } ^ { \prime } \gets a u t o \_ g r a d ( \mathcal { L } ^ { \prime } , \mathbf { x } _ { t } ^ { \prime } )$   
7: Rel $e a s e \_ G r a p h ( f _ { r } ( \mathbf { x } _ { t } ^ { \prime } ) \to \mathbf { x } _ { t - 1 } ^ { \prime } )$ 0   
8: end for   
9: $\partial \mathcal { L } / \partial \mathbf { x } _ { T } \gets \partial \mathcal { L } / \partial \mathbf { x } _ { T } ^ { \prime }$   
10: for $t = T - 1$ to 0 do   
11: Creat $\_ G r a p h ( f _ { d } ( \mathbf x _ { t } ) \to \mathbf x _ { t + 1 } )$   
12: $\mathcal { L } ^ { \prime } \gets ( \partial \mathcal { L } / \partial \mathbf { x } _ { t + 1 } ) \left( f _ { d } ( \mathbf { x } _ { t } ) \right)$   
13: $\partial \mathcal { L } / \partial \mathbf { x } _ { t } \gets a u t o \_ g r a d ( \mathcal { L } ^ { \prime } , \mathbf { x } _ { t } )$   
14: Release_Graph(fd(xt) → xt+1)   
15: end for

# B Efficient Gradient Backpropagation

In this section, we provide the PyTorch-like pseudo-codes of the segment-wise forwardingbackwarding algorithm. At each time step $t$ in the reverse process, we only need to store the gradient $\partial \mathcal { L } / \partial \mathbf { x } _ { t } ^ { \prime }$ , the intermediate sample $\mathbf { x } _ { t + 1 } ^ { \prime }$ and the model $f _ { r }$ to construct the computational graph. When we backpropagate the gradients at the next time step $t + 1$ , the computational graph at time step $t$ will no longer be reused, and thus, we can release the memory of the graph at time step $t$ . Therefore, we only have one segment of the computational graph used for gradient backpropagation in the memory at each time step. We can similarly backpropagate the gradients in the diffusion process.

# C Proofs

# C.1 Proof of Thm. 1

Assumption C.1. Consider adversarial sample $\tilde { \mathbf { x } } _ { 0 } : = \mathbf { x } _ { 0 } + \boldsymbol { \delta }$ , where $\mathbf { x } _ { \mathrm { 0 } }$ is the clean example and $\delta$ is the perturbation. $p _ { t } ( \mathbf { x } ) , p _ { t } ^ { \prime } ( \mathbf { x } ) , q _ { t } ( \mathbf { x } ) , q _ { t } ^ { \prime } ( \mathbf { x } )$ are the distribution of $\mathbf { x } _ { t } , \mathbf { x } _ { t } ^ { \prime } , \tilde { \mathbf { x } } _ { t } , \tilde { \mathbf { x } } _ { t } ^ { \prime }$ where $\mathbf { x } _ { t } ^ { \prime }$ represents the reconstruction of $\mathbf { x } _ { t }$ at time step $t$ in the reverse process. We consider a score-based diffusion model with a well-trained score-based model $\scriptstyle { \pmb { s } } \theta$ parameterized by $\theta$ with the clean training distribution. Therefore, we assume that $\scriptstyle { \pmb { s } } \theta$ can achieve a low score-matching loss given a clean sample and reconstruct it in the reverse process:

$$
\begin{array} { r } { \| \nabla _ { \mathbf { x } } \log p _ { t } ( \mathbf { x } ) - \pmb { s } _ { \theta } ( \mathbf { x } , t ) \| _ { 2 } ^ { 2 } \leq L _ { u } } \\ { D _ { T V } ( p _ { t } , p _ { t } ^ { \prime } ) \leq \epsilon _ { r e } } \end{array}
$$

where $D _ { T V } ( \cdot , \cdot )$ is the total variation distance. $L _ { u }$ and $\epsilon _ { r e }$ are two small constants that characterize the score-matching loss and the reconstruction error.

Assumption C.2. We assume the score function of data distribution is bounded by $M$ :

$$
\| \nabla _ { \mathbf { x } } \log p _ { t } ( \mathbf { x } ) \| _ { 2 } \leq M , \ \| \nabla _ { \mathbf { x } } \log q _ { t } ( \mathbf { x } ) \| _ { 2 } \leq M
$$

Lemma C.1. Consider adversarial sample $\tilde { \mathbf { x } } _ { 0 } : = \mathbf { x } _ { 0 } + \boldsymbol { \delta }$ , where $\mathbf { x } _ { \mathrm { 0 } }$ is the clean example and $\delta$ is the perturbation. $p _ { t } ( \mathbf { x } ) , p _ { t } ^ { \prime } ( \mathbf { x } ) , q _ { t } ( \mathbf { x } ) , q _ { t } ^ { \prime } ( \bar { \mathbf { x } } )$ are the distribution of $\mathbf { x } _ { t } , \mathbf { x } _ { t } ^ { \prime } , \tilde { \mathbf { x } } _ { t } , \tilde { \mathbf { x } } _ { t } ^ { \prime }$ where $\mathbf { x } _ { t } ^ { \prime }$ represents the reconstruction of $\mathbf { x } _ { t }$ in the reverse process. Given a $V P$ -SDE parameterized by $\beta ( \cdot )$ and the score-based model $\scriptstyle { \pmb { s } } \theta$ with Assumption C.2, we have:

$$
D _ { K L } ( p _ { t } ^ { \prime } , q _ { t } ^ { \prime } ) = \frac { 1 } { 2 } \int _ { t } ^ { T } \beta ( s ) \mathbb { E } _ { \mathbf { x } | \mathbf { x } _ { 0 } } \| \nabla _ { \mathbf { x } } \log p _ { s } ^ { \prime } ( \mathbf { x } ) - \nabla _ { \mathbf { x } } \log q _ { s } ^ { \prime } ( \mathbf { x } ) \| _ { 2 } ^ { 2 } d s + 4 M ^ { 2 } \int _ { t } ^ { T } \beta ( s ) d s
$$

Proof. The reverse process of VP-SDE can be formulated as follows:

$$
\begin{array}{c} \begin{array} { r } { \left. \mathbf { x } = f _ { r e v } ( \mathbf { x } , t , p _ { t } ) d t + g _ { r e v } ( t ) d \mathbf { w } , \mathrm { ~ w h e r e ~ } f _ { r e v } ( \mathbf { x } , t , p _ { t } ) = - \cfrac { 1 } { 2 } \beta ( t ) \mathbf { x } - \beta ( t ) \nabla _ { \mathbf { x } } \log p _ { t } ( \mathbf { x } ) , g _ { r e v } ( t ) = \sqrt { \beta ( t ) } , \right.} \end{array}   \end{array}
$$

Using the Fokker-Planck equation [44] in Equation (16), we have:

$$
\begin{array} { l } { \displaystyle \frac { \partial p _ { t } ^ { \prime } ( \mathbf { x } ) } { \partial t } = - \nabla _ { \mathbf { x } } \left( f _ { r e v } ( \mathbf { x } , t , p _ { t } ) p _ { t } ^ { \prime } ( \mathbf { x } ) - \frac { 1 } { 2 } g _ { r e v } ^ { 2 } ( t ) \nabla _ { \mathbf { x } } p _ { t } ^ { \prime } ( x ) \right) } \\ { \displaystyle = \nabla _ { \mathbf { x } } \left( \left( \frac { 1 } { 2 } g _ { r e v } ^ { 2 } ( t ) \nabla _ { \mathbf { x } } \log p _ { t } ^ { \prime } ( \mathbf { x } ) - f _ { r e v } ( \mathbf { x } , t , p _ { t } ) \right) p _ { t } ^ { \prime } ( \mathbf { x } ) \right) } \end{array}
$$

Similarly, applying the Fokker-Planck equation on the reverse SDE for $q _ { t } ^ { \prime } ( \mathbf { x } )$ , we can get:

$$
\frac { \partial q _ { t } ^ { \prime } ( \mathbf { x } ) } { \partial t } = \nabla _ { \mathbf { x } } \left( \left( \frac { 1 } { 2 } g _ { r e v } ^ { 2 } ( t ) \nabla _ { \mathbf { x } } \log q _ { t } ^ { \prime } ( \mathbf { x } ) - f _ { r e v } ( \mathbf { x } , t , q _ { t } ) \right) q _ { t } ^ { \prime } ( \mathbf { x } ) \right)
$$

We use the notation $h _ { p } ( { \bf x } ) \quad = \quad \frac { 1 } { 2 } g _ { r e v } ^ { 2 } ( t ) \nabla _ { \bf x } \log p _ { t } ^ { \prime } ( { \bf x } ) ~ - ~ f _ { r e v } ( { \bf x } , t , p _ { t } ) \quad \mathrm { a n d } \quad h _ { q } ( x \quad { \bf x } , t , x , t )$ $\begin{array} { r l } { h _ { q } ( x ) } & { { } = } \end{array}$ $\frac { 1 } { 2 } g _ { r e v } ^ { 2 } ( t ) \nabla _ { \mathbf { x } } \log q _ { t } ^ { \prime } ( \mathbf { x } ) - f _ { r e v } ( \mathbf { x } , t , q _ { t } )$ . Then according to [34](Theorem A.1), under the assumption that $p _ { t } ^ { \prime } ( \mathbf { x } )$ and $q _ { t } ^ { \prime } ( \mathbf { x } )$ are smooth and fast decaying (i.e., $\begin{array} { r } { \operatorname* { l i m } _ { \mathbf { x } _ { i } \to \infty } [ p _ { t } ^ { \prime } ( \mathbf { x } ) \partial \log p ^ { \prime } ( \mathbf { x } ) / \partial \mathbf { x } _ { i } ] = } \end{array}$ $\begin{array} { r } { 0 , \operatorname* { l i m } _ { \mathbf { x } _ { i } \to \infty } [ q _ { t } ^ { \prime } ( \mathbf { x } ) \partial \log q ^ { \prime } ( \mathbf { x } ) / \partial \mathbf { x } _ { i } ] = 0 , } \end{array}$ ), we have:

$$
\frac { \partial D _ { K L } ( p _ { t } ^ { \prime } , q _ { t } ^ { \prime } ) } { \partial t } = - \int p _ { t } ^ { \prime } ( x ) [ h _ { p } ( \mathbf { x } , t ) - h _ { q } ( \mathbf { x } , t ) ] ^ { T } [ \nabla _ { \mathbf { x } } \log p _ { t } ^ { \prime } ( \mathbf { x } ) - \nabla _ { \mathbf { x } } \log q _ { t } ^ { \prime } ( \mathbf { x } ) ] d \mathbf { x }
$$

Plugging in Equations (18) and (19), we have:

$$
\begin{array} { r l } & { \frac { \partial D _ { K L } ( p _ { t } ^ { \prime } , q _ { t } ^ { \prime } ) } { \partial t } = - \int p _ { t } ^ { \prime } ( \mathbf { x } ) ( \frac { 1 } { 2 } g _ { r e v } ^ { 2 } ( t ) \| \nabla _ { \mathbf { x } } \log p _ { t } ^ { \prime } ( \mathbf { x } ) - \nabla _ { \mathbf { x } } \log q _ { t } ^ { \prime } ( \mathbf { x } ) \| _ { 2 } ^ { 2 } } \\ & { \qquad + \left. \beta ( t ) [ \nabla _ { \mathbf { x } } \log p _ { t } ( \mathbf { x } ) - \nabla _ { \mathbf { x } } \log q _ { t } ( \mathbf { x } ) ] ^ { T } [ \nabla _ { \mathbf { x } } \log p _ { t } ^ { \prime } ( \mathbf { x } ) - \nabla _ { \mathbf { x } } \log q _ { t } ^ { \prime } ( \mathbf { x } ) ] \right) d \mathbf { x } , } \end{array}
$$

Finally, we can derive as follows:

$$
\begin{array} { l } { \displaystyle D _ { K L } ( p _ { t } ^ { \prime } , q _ { t } ^ { \prime } ) = \int _ { t } ^ { T } \int _ { \mathcal { X } } ( p _ { s } ^ { \prime } ( \mathbf { x } ) ( \frac { 1 } { 2 } g _ { r e v } ^ { 2 } ( s ) \| \nabla _ { \mathbf { x } } \log p _ { s } ^ { \prime } ( \mathbf { x } ) - \nabla _ { \mathbf { x } } \log q _ { s } ^ { \prime } ( \mathbf { x } ) \| _ { 2 } ^ { 2 } } \\ { \displaystyle \quad \quad + \beta ( s ) [ \nabla _ { \mathbf { x } } \log p _ { s } ( \mathbf { x } ) - \nabla _ { \mathbf { x } } \log q _ { s } ( \mathbf { x } ) ] ^ { T } [ \nabla _ { \mathbf { x } } \log p _ { s } ^ { \prime } ( \mathbf { x } ) - \nabla _ { \mathbf { x } } \log q _ { s } ^ { \prime } ( \mathbf { x } ) ] ) ) d \mathbf { x } d s } \\ { \displaystyle \quad \quad \leq \int _ { t } ^ { T } ( \frac { 1 } { 2 } g _ { r e v } ^ { 2 } ( s ) \mathbb { E } _ { \mathbf { x } | \mathbf { x } _ { 0 } } \| \nabla _ { \mathbf { x } } \log p _ { s } ^ { \prime } ( \mathbf { x } ) - \nabla _ { \mathbf { x } } \log q _ { s } ^ { \prime } ( \mathbf { x } ) \| _ { 2 } ^ { 2 } + 4 \beta ( s ) M ^ { 2 } ) d s } \\ { \displaystyle \quad = \frac { 1 } { 2 } \int _ { t } ^ { T } \beta ( s ) \mathbb { E } _ { \mathbf { x } | \mathbf { x } _ { 0 } } \| \nabla _ { \mathbf { x } } \log p _ { s } ^ { \prime } ( \mathbf { x } ) - \nabla _ { \mathbf { x } } \log q _ { s } ^ { \prime } ( \mathbf { x } ) \| _ { 2 } ^ { 2 } d s + 4 M ^ { 2 } \int _ { t } ^ { T } \beta ( s ) d s } \end{array}
$$

Theorem 2 (Thm. 1 in the main text). Consider adversarial sample $\tilde { \mathbf { x } } _ { 0 } : = \mathbf { x } _ { 0 } + \delta$ , where $\mathbf { x } _ { \mathrm { 0 } }$ is the clean example and $\delta$ is the perturbation. $p _ { t } ( \mathbf { x } ) , p _ { t } ^ { \prime } ( \mathbf { x } ) , q _ { t } ( \mathbf { x } ) , q _ { t } ^ { \prime } ( \mathbf { x } )$ are the distribution of $\mathbf { x } _ { t } , \mathbf { x } _ { t } ^ { \prime } , \tilde { \mathbf { x } } _ { t } , \tilde { \mathbf { x } } _ { t } ^ { \prime }$ where $\mathbf { x } _ { t } ^ { \prime }$ represents the reconstruction of $\mathbf { x } _ { t }$ in the reverse process. $D _ { T V } ( \cdot , \cdot )$ measures the total variation distance. Given a VP-SDE parameterized by $\beta ( \cdot )$ and the score-based model $\scriptstyle { \pmb { s } } \theta$ with mild assumptions that $\| \nabla _ { \mathbf { x } } \log p _ { t } ( \mathbf { x } ) - s _ { \theta } ( \mathbf { x } , t ) \| _ { 2 } ^ { 2 } \leq L _ { u }$ , $D _ { T V } ( p _ { t } , p _ { t } ^ { \prime } ) \leq \epsilon _ { r e }$ , and a bounded score function by $M$ (specified with details in Appendix C.1), we have:

$$
\begin{array} { c } { \displaystyle D _ { T V } ( q _ { t } , q _ { t } ^ { \prime } ) \leq \frac { 1 } { 2 } \sqrt { \mathbb { E } _ { t , \mathbf { x } | \mathbf { x } _ { 0 } } \| s _ { \theta } ( \mathbf { x } , t ) - \nabla _ { \mathbf { x } } \log q _ { t } ^ { \prime } ( \mathbf { x } ) \| _ { 2 } ^ { 2 } + C _ { 1 } } } \\ { + \sqrt { 2 - 2 \exp \{ - C _ { 2 } \| \delta \| _ { 2 } ^ { 2 } \} } + \epsilon _ { r e } } \\ { { \mathrm {  ~ \sigma ~ } _ { 1 } = ( L _ { u } + 8 M ^ { 2 } ) \int _ { t } ^ { T } \beta ( t ) d t , C _ { 2 } = \frac { 1 } { 8 ( 1 - \Pi _ { s = 1 } ^ { t } ( 1 - \beta _ { s } ) ) } . } } \end{array}
$$

Proof. Since we consider VP-SDE here, we have:

$$
\begin{array} { r l r } & { } & { f ( { \bf x } , t ) = - \displaystyle \frac { 1 } { 2 } \beta ( t ) { \bf x } , \quad g ( t ) = \sqrt { \beta ( t ) } } \\ & { } & { f _ { r e v } ( { \bf x } , t ) = - \displaystyle \frac { 1 } { 2 } \beta ( t ) { \bf x } - \beta ( t ) \nabla _ { \bf x } \log p _ { t } ( { \bf x } ) , \quad g _ { r e v } ( t ) = \sqrt { \beta ( t ) } } \end{array}
$$

Using the triangular inequality, the total variation distance between $q _ { t }$ and $q _ { t } ^ { \prime }$ can be decomposed as:

$$
D _ { T V } ( q _ { t } , q _ { t } ^ { \prime } ) \leq D _ { T V } ( q _ { t } , p _ { t } ) + D _ { T V } ( p _ { t } , p _ { t } ^ { \prime } ) + D _ { T V } ( q _ { t } ^ { \prime } , p _ { t } ^ { \prime } )
$$

According to Assumption C.1, we have $D _ { T V } ( p _ { t } , p _ { t } ^ { \prime } ) \leq \epsilon _ { r e }$ and thus, we only need to upper bound $D _ { T V } ( q _ { t } , \bar { p } _ { t } )$ and $\hat { D _ { T V } } ( q _ { t } ^ { \prime } , p _ { t } ^ { \prime } )$ ,respectively.

Using the notation $\alpha _ { t } : = 1 - \beta ( t )$ and $\bar { \alpha } _ { t } : = \Pi _ { s = 1 } ^ { t } \alpha _ { s }$ , we have:

$$
\mathbf { x } _ { t } \sim p _ { t } : = \mathcal { N } ( \mathbf { x } _ { t } ; \sqrt { \bar { \alpha } _ { t } } \mathbf { x } _ { 0 } , ( 1 - \bar { \alpha } _ { t } ) \mathbf { I } ) , \quad \tilde { \mathbf { x } } _ { t } \sim q _ { t } : = \mathcal { N } ( \tilde { \mathbf { x } } _ { t } ; \sqrt { \bar { \alpha } _ { t } } \tilde { \mathbf { x } } _ { 0 } , ( 1 - \bar { \alpha } _ { t } ) \mathbf { I } )
$$

Therefore, we can upper bound the total variation distance between $q _ { t }$ and $p _ { t }$ as follows:

$$
\begin{array} { l } { \displaystyle D _ { T V } \big ( q _ { t } , p _ { t } \big ) \overset { ( a ) } { \leq } \sqrt { 2 } H ( \mathbf { x } _ { t } , \tilde { \mathbf { x } } _ { t } ) } \\ { \displaystyle \qquad \overset { ( b ) } { = } \sqrt { 2 } \sqrt { 1 - \exp \{ - \frac { 1 } { 8 ( 1 - \bar { \alpha } _ { t } ) } \delta ^ { T } \delta \} } } \\ { \displaystyle \qquad = \sqrt { 2 - 2 \exp \{ - \frac { 1 } { 8 ( 1 - \bar { \alpha } _ { t } ) } \| \delta \| _ { 2 } ^ { 2 } \} } } \end{array}
$$

where we leverage the inequality between the Hellinger distance $H ( \cdot , \cdot )$ and total variation distance in Equation (31)(a) and we plug in the closed form of Hellinger distance between two Gaussian distribution [14] parameterized by $\mu _ { 1 } , \Sigma _ { 1 } , \mu _ { 2 } , \Sigma _ { 2 }$ in Equation (32)(b):

$$
\bar { I } ( \mathcal { N } ( \mu _ { 1 } , \Sigma _ { 1 } ) , \mathcal { N } ( \mu _ { 2 } , \Sigma _ { 2 } ) ) ^ { 2 } = 1 - \frac { d \epsilon t ( \Sigma _ { 1 } ) ^ { 1 / 4 } d \epsilon t ( \Sigma _ { 2 } ) ^ { 1 / 4 } } { d \epsilon t \left( \frac { \Sigma _ { 1 } + \Sigma _ { 2 } } { 2 } \right) ^ { 1 / 2 } } \exp \{ - \frac { 1 } { 8 } ( \mu _ { 1 } - \mu _ { 2 } ) ^ { T } \left( \frac { \Sigma _ { 1 } + \Sigma _ { 2 } } { 2 } \right) ^ { - 1 } ( \mu _ { 1 } - \mu _ { 2 } ) \}
$$

Then, we will upper bound $D _ { T V } ( p _ { t } ^ { \prime } , q _ { t } ^ { \prime } )$ . We first leverage Pinker’s inequality to upper bound the total variation distance with the KL-divergence:

$$
D _ { T V } ( p _ { t } ^ { \prime } , q _ { t } ^ { \prime } ) \leq \sqrt { \frac { 1 } { 2 } D _ { K L } ( p _ { t } ^ { \prime } , q _ { t } ^ { \prime } ) }
$$

Then we plug in the results in Lemma C.1 to upper bound $K L ( p _ { t } ^ { \prime } , q _ { t } ^ { \prime } )$ and it follows that:

$$
\begin{array} { r l r } & { \leq \sqrt { \displaystyle \frac { 1 } { 2 } D _ { K L } ( p _ { t } ^ { \prime } , q _ { t } ^ { \prime } ) } } & { ( 3 \mathcal { T } _ { \mathbf { s } } ) } \\ & { \leq \sqrt { \displaystyle \frac { 1 } { 4 } \int _ { t } ^ { T } \beta ( s ) \mathbb { E } _ { \mathbf { x } | \mathbf { x _ { 0 } } } \| \nabla _ { \mathbf { x } } \log p _ { s } ^ { \prime } ( \mathbf { x } ) - \nabla _ { \mathbf { x } } \log q _ { s } ^ { \prime } ( \mathbf { x } ) \| _ { 2 } ^ { 2 } d s + 2 M ^ { 2 } \int _ { t } ^ { T } \beta ( s ) d s } } & { ( 3 8 \mathcal { T } _ { \mathbf { s } } ) } \\ & { \leq \sqrt { \displaystyle \frac { 1 } { 4 } \int _ { t } ^ { T } \beta ( s ) \mathbb { E } _ { \mathbf { x } | \mathbf { x _ { 0 } } } [ \| \nabla _ { \mathbf { x } } \log p _ { s } ^ { \prime } ( \mathbf { x } ) - s _ { \theta } ( \mathbf { x } , s ) \| _ { 2 } ^ { 2 } + \| s _ { \theta } ( \mathbf { x } , s ) - \nabla _ { \mathbf { x } } \log q _ { s } ^ { \prime } ( \mathbf { x } ) \| _ { 2 } ^ { 2 } ] d s + 2 M ^ { 2 } \int _ { t } ^ { T } \beta ( s ) } } & { ( 3 7 \mathcal { T } _ { \mathbf { s } } ) } \end{array}
$$

$$
\stackrel { ( a ) } { \leq } \sqrt { ( \frac { L _ { u } } { 4 } + 2 M ^ { 2 } ) \int _ { t } ^ { T } \beta ( s ) d s + \frac { 1 } { 4 } \mathbb { E } _ { t , { \bf x } \mid { \bf x } _ { 0 } } \Vert s _ { \theta } ( { \bf x } , t ) - \nabla _ { \bf x } \log q _ { t } ^ { \prime } ( { \bf x } ) \Vert _ { 2 } ^ { 2 } }
$$

where in Equation (40)(a), we leverage the fact that $\beta ( \cdot )$ is bounded in $[ 0 , 1 ]$ .

Combining Equations (29), (33) and (40), we can finally get:

$$
D _ { T V } ( q _ { t } , q _ { t } ^ { \prime } ) \leq \sqrt { \frac { 1 } { 4 } \mathbb { E } _ { t , \mathbf { x } | \mathbf { x } _ { 0 } } \| s _ { \theta } ( \mathbf { x } , t ) - \nabla _ { \mathbf { x } } \log { q _ { t } ^ { \prime } ( \mathbf { x } ) } \| _ { 2 } ^ { 2 } + C _ { 1 } } + \sqrt { 2 - 2 \exp \{ - C _ { 2 } \| \delta \| _ { 2 } ^ { 2 } \} } + \epsilon _ { r e f } ( \log { \epsilon _ { r } ( \mathbf { x } ) } ) - 1
$$

# C.2 Connection between the deviated-reconstruction loss and the density gradient loss for DDPM

Theorem 3. Consider adversarial sample $\tilde { \mathbf { x } } _ { 0 } : = \mathbf { x } _ { 0 } + \delta$ , where $\mathbf { x } _ { \mathrm { 0 } }$ is the clean example and $\delta$ is the perturbation. $p _ { t } ( \mathbf { x } ) , p _ { t } ^ { \prime } ( \mathbf { x } ) , q _ { t } ( \mathbf { x } ) , q _ { t } ^ { \prime } ( \mathbf { x } )$ are the distribution of $\mathbf { x } _ { t } , \mathbf { x } _ { t } ^ { \prime } , \tilde { \mathbf { x } } _ { t } , \tilde { \mathbf { x } } _ { t } ^ { \prime }$ where $\mathbf { x } _ { t } ^ { \prime }$ represents the reconstruction of $\mathbf { x } _ { t }$ in the reverse process. Given a DDPM parameterized by $\beta ( \cdot )$ and the function approximator $\scriptstyle { s _ { \theta } }$ with the mild assumptions that $\| s _ { \theta } ( \mathbf { x } , t ) - \epsilon ( \mathbf { x } _ { t } , t ) \| _ { 2 } ^ { 2 } \leq L _ { u }$ , $\bar { D } _ { T V } ( p _ { t } , p _ { t } ^ { \prime } ) \leq \epsilon _ { r e }$ , and a bounded score function by $M$ (i.e., $\| \epsilon ( \mathbf { x } , t ) \| _ { 2 } \leq M )$ where $\epsilon ( \cdot , \cdot )$ represents the mapping function of the true perturbation, we have:

$$
\begin{array} { r l } & { ( q _ { t } , q _ { t } ^ { \prime } ) \leq \sqrt { 2 - 2 \exp \{ - C _ { 2 } \left( \displaystyle \sum _ { k = t + 1 } ^ { T } \lambda ( k , t ) \| s _ { \theta } ( \tilde { \mathbf { x } } _ { k } ^ { \prime } , k ) - \epsilon ( \tilde { \mathbf { x } } _ { k } ^ { \prime } , k ) \| _ { 2 } + C _ { 1 } \| \delta \| _ { 2 } + ( \sqrt { L _ { u } } + 2 M ) \displaystyle \sum _ { k = t + 1 } ^ { T } \lambda ( k , t ) \right) ^ { 2 } \} } } \\ & { \qquad + \sqrt { 2 - 2 \exp \{ - \frac { 1 } { 8 } ( 1 - \Pi _ { s = 1 } ^ { t } ( 1 - \beta _ { s } ) ) \| \delta \| _ { 2 } ^ { 2 } \} } + \epsilon _ { r e } } \\ & { \mathop { : } \ C _ { 1 } \ = \ \left( \Pi _ { s = t + 1 } ^ { T } \sqrt { \Pi _ { k = 1 } ^ { s } ( 1 - \beta _ { k } ) } \right) \sqrt { \Pi _ { s = 1 } ^ { T } ( 1 - \beta _ { s } ) } , \ C _ { 2 } \ = \ \frac { 1 - \Pi _ { s = 1 } ^ { t } ( 1 - \beta _ { s } ) } { 8 ( 1 - \Pi _ { s = 1 } ^ { t - 1 } ( 1 - \beta _ { s } ) ) \beta _ { t } } , } \\ & { \mathrm { ) } = \frac { \beta _ { k } \Pi _ { s = t + 1 } ^ { k - 1 } \sqrt { \Pi _ { s = 1 } ^ { i } ( 1 - \beta _ { s } ) } } { \sqrt { 1 - \Pi _ { s = 1 } ^ { k } ( 1 - \beta _ { s } ) } } . } \end{array}
$$

Proof. For ease of notation, we use the notation: $\alpha _ { t } : = 1 - \beta _ { t }$ and $\bar { \alpha } _ { t } : = \Pi _ { s = 1 } ^ { t } \alpha _ { s }$ . From the DDPM sampling process [25], we know that:

$$
\begin{array} { r l } & { \mathbf { x } _ { t - 1 } ^ { \prime } \sim p _ { t } ^ { \prime } : = \cfrac { 1 } { \sqrt { \bar { \alpha } _ { t } } } \left( \mathbf { x } _ { t } ^ { \prime } - \cfrac { 1 - \alpha _ { t } } { \sqrt { 1 - \bar { \alpha } _ { t } } } s _ { \theta } ( \mathbf { x } _ { t } ^ { \prime } , t ) \right) + \sigma _ { t } \mathbf { z } } \\ & { \tilde { \mathbf { x } } _ { t - 1 } ^ { \prime } \sim q _ { t } ^ { \prime } : = \cfrac { 1 } { \sqrt { \bar { \alpha } _ { t } } } \left( \tilde { \mathbf { x } } _ { t } ^ { \prime } - \cfrac { 1 - \alpha _ { t } } { \sqrt { 1 - \bar { \alpha } _ { t } } } s _ { \theta } ( \tilde { \mathbf { x } } _ { t } ^ { \prime } , t ) \right) + \sigma _ { t } \mathbf { z } } \end{array}
$$

where $\sigma _ { t } ^ { 2 } = \frac { 1 - \bar { \alpha } _ { t - 1 } } { 1 - \bar { \alpha } _ { t } } \beta _ { t }$

$\mu _ { t , q }$ and $\mu _ { t , p }$ represent the mean of the distribution $q _ { t } ^ { \prime }$ and $p _ { t } ^ { \prime }$ , respectively. Then from Equations (43) and (44), we have:

$$
\mu _ { t , q } - \mu _ { t , p } = \frac { 1 } { \sqrt { \bar { \alpha _ { t } } } } ( \mu _ { t - 1 , q } - \mu _ { t - 1 , p } ) - \frac { 1 - \alpha _ { t } } { \sqrt { \bar { \alpha _ { t } } } \sqrt { 1 - \bar { \alpha } _ { t } } } ( s _ { \theta } ( \tilde { \mathbf { x } } _ { t } ^ { \prime } , t ) - s _ { \theta } ( \mathbf { x } _ { t } ^ { \prime } , t ) )
$$

Applying Equation (45) iteratively, we get:

$$
\mu _ { T , q } - \mu _ { T , p } = \frac { 1 } { \prod _ { s = t } ^ { T } \sqrt { \bar { \alpha } _ { s } } } \big ( \mu _ { t - 1 , q } - \mu _ { t - 1 , p } \big ) - \sum _ { k = t } ^ { T } \frac { 1 - \alpha _ { k } } { \sqrt { 1 - \bar { \alpha } _ { k } } \prod _ { i = k } ^ { T } \sqrt { \bar { \alpha } _ { i } } } \big ( s \theta ( \tilde { \mathbf { x } } _ { k } ^ { \prime } , k ) - s _ { \theta } ( \mathbf { x } _ { k } ^ { \prime } , k ) \big )
$$

On the other hand, $\mu _ { T , q } - \mu _ { T , p }$ can be formulated explicitly considering the Gaussian distribution at time step $T$ in the diffusion process:

$$
\pmb { \mu } _ { T , q } - \pmb { \mu } _ { T , p } = \sqrt { \bar { \alpha } _ { T } } ( \tilde { \bf x } _ { 0 } - { \bf x } _ { 0 } ) = \sqrt { \bar { \alpha } _ { T } } \delta
$$

Combining Equations (46) and (47), we can derive that:

$$
\begin{array} { r l } & { \quad \| \mu _ { T , q } - \mu _ { T , p } \| _ { 2 } } \\ & { = ( \Pi _ { s = t + 1 } ^ { T } \sqrt { \bar { \alpha } _ { s } } ) \sqrt { \bar { \alpha } _ { T } } \| \delta \| _ { 2 } + \displaystyle \sum _ { k = t + 1 } ^ { T } \frac { ( 1 - \alpha _ { k } ) \Pi _ { i = t + 1 } ^ { k - 1 } \sqrt { \bar { \alpha } _ { i } } } { \sqrt { 1 - \bar { \alpha } _ { k } } } \| s _ { \theta } ( \bar { \mathbf { x } } _ { k } ^ { \prime } , k ) - s _ { \theta } ( \mathbf { x } _ { k } ^ { \prime } , k ) ) \| _ { 2 } } \\ & { \leq ( \Pi _ { s = t + 1 } ^ { T } \sqrt { \bar { \alpha } _ { s } } ) \sqrt { \bar { \alpha } _ { T } } \| \delta \| _ { 2 } + \displaystyle \sum _ { k = t + 1 } ^ { T } \frac { ( 1 - \alpha _ { k } ) \Pi _ { i = t + 1 } ^ { k - 1 } \sqrt { \bar { \alpha } _ { i } } } { \sqrt { 1 - \bar { \alpha } _ { k } } } ( \| s _ { \theta } ( \bar { \mathbf { x } } _ { k } ^ { \prime } , k ) - \epsilon ( \mathbf { x } _ { k } ^ { \prime } , k ) \| _ { 2 } + \| \epsilon ( \mathbf { x } _ { k } ^ { \prime } , k ) - s _ { \theta } ( \mathbf { x } _ { k } ^ { \prime } , k ) ) \| _ { 2 } ) } \\ &  \leq ( \Pi _ { s = t + 1 } ^ { T } \sqrt { \bar { \alpha } _ { s } } ) \sqrt { \bar { \alpha } _ { T } } \| \delta \| _ { 2 } + \sqrt { L _ { u } } \displaystyle \sum _ { k = t + 1 } ^ { T } \lambda ( k , t ) + \displaystyle \sum _ { k = t + 1 } ^ { T } \lambda ( k , t ) ( \| s _ { \theta } ( \tilde { \mathbf { x } } _ { k } ^ { \prime } , k ) - \epsilon ( \tilde { \mathbf { x } } _ { k } ^   \end{array}
$$

$$
\leq \left( \prod _ { s = t + 1 } ^ { T } \sqrt { \bar { \alpha } _ { s } } \right) \sqrt { \bar { \alpha } _ { T } } \| \delta \| _ { 2 } + ( \sqrt { L _ { u } } + 2 M ) \sum _ { k = t + 1 } ^ { T } \lambda ( k , t ) + \sum _ { k = t + 1 } ^ { T } \lambda ( k , t ) \| s _ { \theta } ( \tilde { \mathbf { x } } _ { k } ^ { \prime } , k ) - \epsilon ( \tilde { \mathbf { x } } _ { k } ^ { \prime } , k ) \| _ { 2 }
$$

(1 − αk)Πk−1i=t+1√α¯i

We then leverage the closed form formulation of the Hellinger distance between two Gaussian distributions [14] parameterized by $\mu _ { 1 } , \Sigma _ { 1 } , \mu _ { 2 } , \Sigma _ { 2 }$ :

$$
{ \cal T } ^ { 2 } ( { \cal N } ( \mu _ { 1 } , \Sigma _ { 1 } ) , { \cal N } ( \mu _ { 2 } , \Sigma _ { 2 } ) ) = 1 - { \frac { d e t ( \Sigma _ { 1 } ) ^ { 1 / 4 } d e t ( \Sigma _ { 2 } ) ^ { 1 / 4 } } { d e t \left( { \frac { \Sigma _ { 1 } + \Sigma _ { 2 } } { 2 } } \right) ^ { 1 / 2 } } } \exp \{ - { \frac { 1 } { 8 } } ( \mu _ { 1 } - \mu _ { 2 } ) ^ { T } \left( { \frac { \Sigma _ { 1 } + \Sigma _ { 2 } } { 2 } } \right) ^ { - 1 } ( \mu _ { 1 } - \mu _ { 2 } ) \}
$$

Applying it to distribution $p _ { t } ^ { \prime }$ and $q _ { t } ^ { \prime }$ , we have:

$$
H ^ { 2 } ( p _ { t } ^ { \prime } , q _ { t } ^ { \prime } ) = 1 - \exp \{ - \frac { 1 - \bar { \alpha } _ { t } } { 8 ( 1 - \bar { \alpha } _ { t - 1 } ) \beta _ { t } } \| \mu _ { t , q } - \mu _ { t , p } \| _ { 2 } ^ { 2 } \}
$$

$$
\leq 1 - \exp \{ - C _ { 2 } \left( C _ { 1 } \| \delta \| _ { 2 } + ( \sqrt { L _ { u } } + 2 M ) \sum _ { k = t + 1 } ^ { T } \lambda ( k , t ) + \sum _ { k = t + 1 } ^ { T } \lambda ( k , t ) \| s _ { \theta } ( \tilde { \mathbf { x } } _ { k } ^ { \prime } , k ) - \epsilon ( \tilde { \mathbf { x } } _ { k } ^ { \prime } , k ) \| _ { 2 } \right) ^ { 2 } \}
$$

where C1 =  ΠTs=t+1√α¯s √α¯T and C2 = 1 − α¯t8(1  α¯t−1)βt . Finally, it follows that:

$$
\begin{array} { r l r } {  { D _ { T V } ( q _ { t } , q _ { t } ^ { \prime } ) \leq D _ { T V } ( q _ { t } , p _ { t } ) + D _ { T V } ( p _ { t } , p _ { t } ^ { \prime } ) + D _ { T V } ( q _ { t } ^ { \prime } , p _ { t } ^ { \prime } ) } } & { \mathrm { ( s u p ~ } ( \boldsymbol { q } _ { t } ^ { \prime } , \boldsymbol { p } _ { t } ^ { \prime } ) } \\ & { } & { \leq \sqrt { 2 - 2 \exp \{ - \frac { 1 } { 8 } ( 1 - \bar { \alpha } _ { t } ) \| \delta \| _ { 2 } ^ { 2 } \} } + \epsilon _ { r e } + \sqrt { 2 } H ( q _ { t } ^ { \prime } , p _ { t } ^ { \prime } ) ~ ( \boldsymbol { s } ; \boldsymbol { q } _ { t } ^ { \prime } ) ~ ( \boldsymbol { s } ; \boldsymbol { p } _ { t } ^ { \prime } ) ~ ( \boldsymbol { s } ; \boldsymbol { p } _ { t } ^ { \prime } ) ~ ( \boldsymbol { s } ; \boldsymbol { p } _ { t } ^ { \prime } ) ~ ( \boldsymbol { s } ; \boldsymbol { p } _ { t } ^ { \prime } ) ~ ( \boldsymbol { s } ; \boldsymbol { p } _ { t } ^ { \prime } ) } \\ & { } & { \leq \sqrt { 2 - 2 \exp \{ - C _ { 2 } ( C _ { 1 } \| \delta \| _ { 2 } + ( \sqrt { L _ { u } } + 2 M ) \sum _ { k = t + 1 } ^ { T } \lambda ( k , t ) + \sum _ { k = t + 1 } ^ { T } \lambda ( k , t ) \| s _ { \theta } ( \tilde { \mathbf { x } } _ { k } ^ { \prime } , k ) - \epsilon ( \tilde { \mathbf { x } } _ { k } ^ { \prime } , k ) \| _ { 2 } ) ^ { 2 } \} } ~ } \end{array}
$$

$$
\begin{array} { l } { { { \displaystyle \quad + \sqrt { 2 - 2 \exp \{ - \frac { 1 } { 8 } ( 1 - \bar { \alpha } _ { t } ) \| \delta \| _ { 2 } ^ { 2 } \} } + \epsilon _ { r e } } \qquad } } & { { \displaystyle ( 5 9 ) } } \\ { { { \displaystyle = \sqrt { 2 - 2 \exp \{ - C _ { 2 } \left( \sum _ { k = t + 1 } ^ { T } \lambda ( k , t ) \| s _ { \theta } ( \tilde { \mathbf { x } } _ { k } ^ { \prime } , k ) - \epsilon ( \tilde { \mathbf { x } } _ { k } ^ { \prime } , k ) \| _ { 2 } + C _ { 1 } \| \delta \| _ { 2 } + ( \sqrt { L _ { u } } + 2 M ) \sum _ { k = t + 1 } ^ { T } \lambda ( k , t ) \right) ^ { 2 } \} + \epsilon _ { r e } } \exp \{ - \sum _ { k = t + 1 } ^ { T } \lambda ( k , t ) \} } } } \end{array}
$$

$$
+ \sqrt { 2 - 2 \exp \{ - \frac { 1 } { 8 } ( 1 - \Pi _ { s = 1 } ^ { t } ( 1 - \beta _ { s } ) ) \| \delta \| _ { 2 } ^ { 2 } \} } + \epsilon _ { r e }
$$

$$
\lambda ( k , t ) = \frac { \beta _ { k } \Pi _ { i = t + 1 } ^ { k - 1 } \sqrt { \Pi _ { s = 1 } ^ { i } ( 1 - \beta _ { s } ) } } { \sqrt { 1 - \Pi _ { s = 1 } ^ { k } ( 1 - \beta _ { s } ) } } .
$$

# D Experiment

# D.1 Pseudo-code of DiffAttack

Given an input pair $\left( \mathbf { x } , y \right)$ and the perturbation budget, we notate $\mathcal { L } : = \mathcal { L } _ { c l s } + \lambda \mathcal { L } _ { d e v }$ (Equation (8)) the surrogate loss, $\Pi$ the projection operator given the perturbation budget and distance metric, $\eta$ the step size, $\alpha$ the momentum coefficient, $N _ { \mathrm { i t e r } }$ the number of iterations, and $W$ the set of checkpoint iterations. Concretely, we select $\mathcal { L } _ { c l s }$ as the cross-entropy loss in the first round and DLR loss in the second round following [34]. The gradient of the surrogate loss with respect to the samples is computed by forwarding the samples and backwarding the gradients for multiple times and taking the average to tackle the problem of randomness. We also optimize all the samples, including the misclassified ones, to push them away from the decision boundary. The gradient can be computed with our segment-wise forwarding-backwarding algorithm in Section 3.3, which enables DiffAttack to be the first fully adaptive attack against the DDPM-based purification defense. The complete procedure is provided in Algorithm 2.

# D.2 Experiment details

We use pretrained score-based diffusion models [49] on CIFAR-10, guided diffusion models [15] on ImageNet, and DDPM [25] on CIFAR-10 to purify the images following the literature [34, 4, 55, 56]. Due to the high computational cost, we follow [34] to randomly select a fixed subset of 512 images sampled from the test set to evaluate the robust accuracy for fair comparisons. We implement DiffAttack in the framework of AutoAttack [12], and we use the same hyperparameters. Specifically, the number of iterations of attacks $( N _ { \mathrm { i t e r } } )$ is 100, and the number of iterations to approximate the gradients (EOT) is 20. The momentum coefficient $\alpha$ is 0.75, and the step size $\eta$ is initialized with $2 \epsilon$ where $\epsilon$ is the maximum $\ell _ { p }$ -norm of the perturbations. The balance factor $\lambda$ between the classificationguided loss and the deviated-reconstruction loss in Equation (8) is fixed as 1.0 and $\alpha ( \cdot )$ is set the reciprocal of the size of sampled time steps in the evaluation. We consider $\epsilon = 8 / 2 5 5$ and $\epsilon = 4 / 2 5 5$ for $\ell _ { \infty }$ attack and $\epsilon = 0 . 5$ for $\ell _ { 2 }$ attack following the literature [11, 12].

We use randomly selected 3 seeds and report the averaged results for evaluations. CIFAR-10 is under the MIT license and ImageNet is under the BSD 3-clause license.

More details of baselines. In this part, we illustrate more details of the baselines 1) SOTA attacks against score-based diffusion adjoint attack (AdjAttack) [34], 2) SOTA attack against DDPM-based

# Algorithm 2 DiffAttack

1: Input: $\mathcal { L } : = \mathcal { L } _ { c l s } + \lambda \mathcal { L } _ { d e v } , \Pi , ( { \bf x } , y ) , \eta , \alpha , N _ { \mathrm { i t e r } } , W = \left\{ w _ { 0 } , \dots , w _ { n } \right\}$   
2: Output: x˜   
3: $\tilde { \mathbf { x } } ^ { ( 0 ) }  \tilde { \mathbf { x } }$   
4: $\tilde { \mathbf { x } } ^ { ( 1 ) } \gets \Pi \left( \tilde { \mathbf { x } } ^ { ( 0 ) } + \eta \nabla _ { \tilde { \mathbf { x } } ^ { ( 0 ) } } \mathcal { L } ( \tilde { \mathbf { x } } ^ { ( 0 ) } , y ) \right)$   
5: $L _ { \mathrm { m a x } } \gets \mathrm { m a x } \{ \mathcal { L } ( \tilde { \mathbf { x } } ^ { ( 0 ) } , y ) , \mathcal { L } ( \tilde { \mathbf { x } } ^ { ( 1 ) } , y ) \}$   
6: $\tilde { \mathbf { x } } \gets \tilde { \mathbf { x } } ^ { ( 0 ) }$ if $L _ { \mathrm { m a x } } \equiv \mathcal { L } ( \tilde { \mathbf { x } } ^ { ( 0 ) } , y )$ else $\tilde { \mathbf { x } }  \tilde { \mathbf { x } } ^ { ( 1 ) }$   
7: for $k = 1$ to $N _ { \mathrm { i t e r } } { - 1 }$ do   
8: $\begin{array} { r l } & { \mathbf { z } ^ { ( k + 1 ) }  \Pi ( \tilde { \mathbf { x } } ^ { ( k ) } + \eta \nabla _ { \tilde { \mathbf { x } } ^ { ( k ) } } \mathcal { L } ( \tilde { \mathbf { x } } ^ { ( k ) } , y ) ) } \\ & { \tilde { \mathbf { x } } ^ { ( k + 1 ) }  \Pi ( \tilde { \mathbf { x } } ^ { ( k ) } + \alpha ( \mathbf { z } ^ { ( k + 1 ) } - \tilde { \mathbf { x } } ^ { ( k ) } ) + ( 1 - \alpha ) ( \tilde { \mathbf { x } } ^ { ( k ) } - \tilde { \mathbf { x } } ^ { ( k - 1 ) } ) ) } \end{array}$   
9:   
10: if $\mathcal { L } ( \tilde { \mathbf { x } } ^ { ( k + 1 ) } , y ) > L _ { \mathrm { m a x } }$ then   
11: $\tilde { \mathbf { x } }  \tilde { \mathbf { x } } ^ { ( k + 1 ) }$ and $L _ { \operatorname* { m a x } }  \mathcal { L } ( \tilde { \mathbf { x } } ^ { ( k + 1 ) } , y )$   
12: end if   
13: if $k \in W$ then   
14: $\eta  \eta / 2$   
15: end if   
16: end for

diffusion Diff-BPDA attack [4], 3) SOTA black-box attack SPSA [53] and square attack [1], and 4) specific attack against EBM-based purification joint attack (score/full) [60]. AdjAttack selects the surrogate loss $\mathcal { L }$ as the cross-entropy loss and DLR loss and leverages the adjoint method [28] to efficiently backpropagate the gradients through SDE. The basic idea is to obtain the gradients via solving an augmented SDE. For the SDE in Equation (4), the augmented SDE that computes the gradients ${ \partial \mathcal { L } } / { \partial { \bf x } _ { T } ^ { \prime } }$ of back=propagating through it is given by:

$$
\left( \begin{array} { c } { { { \bf x } _ { T } ^ { \prime } } } \\ { { \frac { \partial \mathcal { L } } { \partial { \bf x } _ { T } ^ { \prime } } } } \end{array} \right) = \mathrm { s d e i n t } \left( \left( \begin{array} { c } { { { \bf x } _ { 0 } ^ { \prime } } } \\ { { \frac { \partial \mathcal { L } } { \partial { \bf x } _ { 0 } ^ { \prime } } } } \end{array} \right) , \tilde { { \bf f } } , \tilde { { \bf g } } , \tilde { { \bf w } } , 0 , T \right)
$$

where $\frac { \partial \mathcal { L } } { \partial \mathbf { x } _ { 0 } ^ { \prime } }$ is the gradient of the objective $\mathcal { L }$ w.r.t. the $\mathbf { x } _ { \mathrm { 0 } } ^ { \prime }$ , and

$$
\begin{array} { r l r } & { } & { \tilde { \bf f } ( [ { \bf x } ; { \bf z } ] , t ) = \left( \begin{array} { c } { { \bf f } _ { \mathrm { r e v } } ( { \bf x } , t ) } \\ { \frac { \partial { \bf f } _ { \mathrm { r e v } } ( { \bf x } , t ) } { \partial { \bf x } } { \bf z } } \end{array} \right) } \\ & { } & { \tilde { \bf g } ( t ) = \left( \begin{array} { c } { - { \bf g } _ { \mathrm { r e v } } ( t ) { \bf 1 } _ { d } } \\ { { \bf 0 } _ { d } } \end{array} \right) } \\ & { } & { \tilde { \bf w } ( t ) = \left( \begin{array} { c } { - { \bf w } ( 1 - t ) } \\ { - { \bf w } ( 1 - t ) } \end{array} \right) } \end{array}
$$

with $\mathbf { 1 } _ { d }$ and $\mathbf { 0 } _ { d }$ representing the $d$ -dimensional vectors of all ones and all zeros, respectively and $\begin{array} { r } { \mathbf { f } _ { \mathrm { r e v } } ( \mathbf { x } , t ) : = - \frac { 1 } { 2 } \beta ( t ) \mathbf { x } - \beta ( t ) \nabla _ { \mathbf { x } } \log p _ { t } ( \mathbf { x } ) , \mathbf { g } _ { \mathrm { r e v } } ( t ) : = \sqrt { \beta ( t ) } . } \end{array}$ .

SPSA attack approximates the gradients by randomly sampling from a pre-defined distribution and using the finite-difference method. Square attack heuristically searches for adversarial examples in a low-dimensional space with the constraints of the perturbation pattern (i.e., constraining the square shape of the perturbation). Joint attack (score) updates the input by the weighted average of the classifier gradient and the output of the score estimation network (i.e., the gradient of log-likelihood with respect to the input), while joint attack (full) leverages the classifier gradients and the difference between the input and the purified samples. The update of the joint attack (score) is formulated as follows:

$$
\tilde { \mathbf { x } } \gets \tilde { \mathbf { x } } + \eta ( \lambda ^ { \prime } \mathbf { s i g n } ( s _ { \theta } ( \tilde { \mathbf { x } } ) ) + ( 1 - \lambda ^ { \prime } ) \mathbf { s i g n } ( \nabla _ { \tilde { \mathbf { x } } } \mathcal { L } ( F ( P ( \tilde { \mathbf { x } } ) ) , y ) )
$$

The update of the joint attack (full) is formulated as follows:

$$
\tilde { \mathbf { x } } \gets \tilde { \mathbf { x } } + \eta \left( \lambda ^ { \prime } \mathbf { s i g n } ( F ( P ( \tilde { \mathbf { x } } ) ) - \tilde { \mathbf { x } } ) + ( 1 - \lambda ^ { \prime } ) \mathbf { s i g n } ( \nabla _ { \tilde { \mathbf { x } } } \mathcal { L } ( F ( P ( \tilde { \mathbf { x } } ) ) , y ) ) \right)
$$

where $\eta$ is the step size and $\lambda ^ { \prime }$ the balance factor fixed as 0.5 in the evaluation.

Table 5: Comparisons of gradient backpropagation time per batch(second)/Memory cost (MB) between [4] and DiffAttack. We evaluate on CIFAR-10 with WideResNet-28-10 with batch size 16.   

<table><tr><td>Method</td><td>T=5</td><td>T=10</td><td>T=15</td><td>T=20</td><td>T=30</td><td>T=1000</td></tr><tr><td>[4]</td><td>0.45/14,491</td><td>0.83/23,735</td><td>1.25/32,905</td><td>1.80/38,771</td><td></td><td></td></tr><tr><td>DiffAttack</td><td>0.44/2,773</td><td>0.85/2,731</td><td>1.26/2,805</td><td>1.82/2,819</td><td>2.67/2,884</td><td>85.81/3,941</td></tr></table>

Table 6: The clean / robust accuracy $( \% )$ of diffusion-based purification with different diffusion lengths $T$ under DiffAttack. The evaluation is done on ImageNet with ResNet-50 under $\ell _ { \infty }$ attack $\check { \epsilon } \doteq \dot { 4 } / 2 5 5 )$ .   

<table><tr><td>T=50</td><td>T=100</td><td>T=150</td><td>T= 200</td></tr><tr><td>71.88 / 12.46</td><td>68.75 / 24.62</td><td>67.79 / 28.13</td><td>65.62 /26.83</td></tr></table>

# D.3 Additional Experiment Results

Efficiency evaluation. We evaluate the wall clock time per gradient backpropagation of the segmentwise forwarding-backwarding algorithm for different diffusion lengths and compare the time efficiency as well as the memory costs with the standard gradient backpropagation in previous attacks [4]. The results in Table 5 indicate that the segment-wise forwarding-backwarding algorithm consumes comparable wall clock time per gradient backpropagation compared with [4] and achieves a much better tradeoff between time efficiency and memory efficiency. The evaluation is done on an RTX A6000 GPU with $^ { 4 9 , 1 4 0 \mathrm { M B } }$ memory. In the segment-wise forwarding-backwarding algorithm, we require one forward pass and one backpropagation pass in total for the gradient computation, while the standard gradient backpropagation in [4] requires one backpropagation pass. However, since the backpropagation pass is much more expensive than the forward pass [36], our segmentwise forwarding-backwarding algorithm can achieve comparable time efficiency while significantly reducing memory costs.

More ablation studies on ImageNet. We conduct more evaluations on ImageNet to consolidate the findings in CIFAR-10. We evaluate the clean/robust accuracy $( \% )$ of diffusion-based purification with different diffusion lengths $T$ under DiffAttack. The results in Table 6 indicate that 1) the clean accuracy of the purification defenses negatively correlates with the diffusion lengths, and 2) a moderate diffusion length benefits the robust accuracy under DiffAttack.

Tansferability of DiffAttack. ACA [10] and Diff-PGD attack [58] explore the transferability of unrestricted adversarial attack, which generates realistic adversarial examples to fool the classifier and maintain the photorealism. They demonstrate that this kind of semantic attack transfers well to other models. To explore the transferability of adversarial examples by $\ell _ { p }$ -norm-based DiffAttack, we evaluate the adversarial examples generated on score-based purification withResNet-50 on defenses with pretrained WRN-50-2 and DeiT-S. The results in Table 7 indicate that DiffAttack also transfers better than AdjAttack and achieves much lower robust accuracy on other models.

Ablation study of balance factor $\lambda$ . As shown in Equation (11), $\lambda$ controls the balance of the two objectives. A small $\lambda$ can weaken the deviated-reconstruction object and make the attack suffer more from the vanishing/exploded gradient problem, while a large $\lambda$ can downplay the guidance of the classification loss and confuse the direction towards the decision boundary of the classifier. The results in Table 8 show that selecting $\lambda$ as 1.0 achieves better tradeoffs empirically, so we fix it as 1.0 for experiments.

# D.4 Visualization

In this section, we provide the visualization of adversarial examples generated by DiffAttack. Based on the visualization on CIFAR-10 and ImageNet with different network architectures, we conclude that the perturbation generated by DiffAttack is stealthy and imperceptible to human eyes and hard to be utilized by defenses.

Table 7: Robust accuracy $( \% )$ with $\ell _ { \infty }$ attack $\zeta = 8 / 2 5 5 )$ ) against score-based diffusion purification on CIFAR-10. The adversarial examples are optimized on the diffusion purification defense with pretrained ResNet-50 and evaluated on defenses with other types of models including WRN-50-2 and DeiT-S.   

<table><tr><td></td><td>ResNet-50</td><td>WRN-50-2</td><td>DeiT-S</td></tr><tr><td>AdjAttack</td><td>40.93</td><td>52.37</td><td>54.53</td></tr><tr><td>DiffAttack</td><td>28.13</td><td>37.28</td><td>39.62</td></tr></table>

Table 8: The impact of different loss weights $\lambda$ on the robust accuracy $( \% )$ . We perform $\ell _ { \infty }$ $( \epsilon = 8 / 2 5 5 )$ against score-based diffusion purification on CIFAR-10 with WideResNet-28-10 and diffusion length $T = 0 . 1$ .   

<table><tr><td>入= 0.1</td><td>入=1.0</td><td>入= 10.0</td></tr><tr><td>54.69</td><td>46.88</td><td>53.12</td></tr></table>

![](images/f9e418067389732a51dc7300d7c95150f5e0f915b4e8b6698c44bb3c9709859a.jpg)  
Figure 5: Visualization of the clean images and adversarial samples generated by DiffAttack on CIFAR-10 with $\ell _ { \infty }$ attack $\zeta = 8 / 2 5 5 )$ ) against score-based purification with WideResNet-28-10.

![](images/1cb6e88fd428ec100024109953c4e6556b951b28d13e51dfad3b29b8bbcbc9f8.jpg)  
Figure 6: Visualization of the clean images and adversarial samples generated by DiffAttack on CIFAR-10 with $\ell _ { \infty }$ attack $\zeta = 8 / 2 5 5 )$ ) against score-based purification with WideResNet-70-16.

![](images/a5044554706e83a6ef3a5efb3642faaccd735820b7d92f75ffdca06ebc27be3c.jpg)  
Figure 7: Visualization of the clean images and adversarial samples generated by DiffAttack on ImageNet with $\ell _ { \infty }$ attack $( \epsilon = 4 / 2 5 5 )$ against score-based purification with WideResNet-50-2.

![](images/abb557c601104cf8496a88d1ebe4ae683dbdf8f4fda9521772ddc90164117851.jpg)  
Figure 8: Visualization of the clean images and adversarial samples generated by DiffAttack on ImageNet with $\ell _ { \infty }$ attack $( \epsilon = 4 / 2 5 5 )$ against score-based purification with DeiT-S.

![](images/bc2cb90fe20a3825d582b2c8fb192e6d3f4059be56680a9afbc9c17d6cb42037.jpg)  
Figure 9: Visualization of the clean images and adversarial samples generated by DiffAttack on ImageNet with a larger perturbation radius: $\ell _ { \infty }$ attack $\epsilon = 8 / 2 5 5 )$ ) against score-based purification with ResNet-50.