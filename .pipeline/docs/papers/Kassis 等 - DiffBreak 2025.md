
# DiffBreak: Is Diffusion-Based Purification Robust?

Andre Kassis, Urs Hengartner, Yaoliang Yu

Cheriton School of Computer Science

University of Waterloo

Waterloo, Ontario, Canada

{akassis, urs.hengartner, yaoliang.yu}@uwaterloo.ca

## Abstract

Diffusion-based purification (DBP) has become a cornerstone defense against adversarial examples (AEs), regarded as robust due to its use of diffusion models (DMs) that project AEs onto the natural data manifold. We refute this core claim, theoretically proving that gradient-based attacks effectively target the DM rather than the classifier, causing DBP’s outputs to align with adversarial distributions. This prompts a reassessment of DBP’s robustness, accrediting it two critical flaws: incorrect gradients and inappropriate evaluation protocols that test only a single random purification of the AE. We show that with proper accounting for stochasticity and resubmission risk, DBP collapses. To support this, we introduce DiffBreak, the first reliable toolkit for differentiation through DBP, eliminating gradient flaws that previously further inflated robustness estimates. We also analyze the current defense scheme used for DBP where classification relies on a single purification, pinpointing its inherent invalidity. We provide a statistically grounded majorityvote (MV) alternative that aggregates predictions across multiple purified copies, showing partial but meaningful robustness gain. We then propose a novel adaptation of an optimization method against deepfake watermarking, crafting systemic perturbations that defeat DBP even under MV, challenging DBP’s viability.

## 1 Introduction

ML classifiers are vulnerable to Adversarial Examples (AEs)—imperceptible perturbations that induce misclassification [37, 15]. Adversarial Training [28, 54] is attack-specific and costly [44], while other defenses [16, 13, 3, 46, 49, 32, 50, 9] are vulnerable to adaptive attacks [38]. Diffusion-Based Purification (DBP) [30, 40], which leverages diffusion models (DMs) [36, 34], has emerged as a promising defense [6, 45, 7, 53, 31, 42]. DBP purifies inputs by solving the reverse stochastic differential equation (SDE) associated with DMs, diffusing the input with random noise to dilute adversarial changes and iteratively denoising it. Given DMs’ data modeling abilities [12, 39], DBP operates under the assumption that each denoising (reverse) step’s output belongs to a corresponding marginal natural distribution (a distribution obtained by introducing Gaussian noise into natural inputs, with decreasing variance proportional to that step).

This assumption is the foundation for DBP’s robustness as it ensures its outputs are from the natural data distribution, while AEs lie outside this manifold and are unlikely to be preserved by DBP. Recent works treat DBP as a static pre-processing step that standard gradient-based attacks cannot directly manipulate but rather only attempt to evade. Consequently, attackers are assumed to either fail due to randomness that masks the classifier’s vulnerabilities [26], or be forced to rely on surrogate losses and heuristic approximations [20, 41]. We theoretically refute DBP’s robustness hypothesis rooted in its ability to perform stochastic projection onto the natural manifold. This is paradoxical as it assumes correct behavior of the score model sθ used by the DM to mimic the gradients of the marginal distributions. Yet, sθ itself is an ML system with exploitable imperfections. In §3.1, we prove that standard adaptive attacks that simply backpropagate the classifier loss gradients through DBP effectively target $\scriptstyle { \pmb { s } } \theta$ rather than the classifier, forcing it to generate samples from an adversarial distribution. Hence, $D B P ^ { \circ } { \bf s }$ theory no longer holds. Worse yet, this implies that previous enhancements [41, 20] are unnecessary at best and often counterproductive, as we prove empirically.

<!-- image-->  
Figure 1: DiffGrad. x is given to the iterative attack algorithm $\scriptstyle A _ { \theta }$ to generate $\pmb { x } _ { a d v }$ . At each iteration, $\pmb { x } _ { a d v }$ is propagated through DBP, yielding $\hat { \pmb x } _ { a d v } ( 0 )$ that is given to M while storing each intermediate $\hat { \pmb x } _ { a d v } ( t )$ (bottom replicas) from $D B P ^ { \prime } { \bf s }$ reverse pass (see §2) but without saving any graph dependencies. Backpropagation uses the stored samples: Starting from t“´dt (dt ă 0—see §2), each $\hat { \pmb x } _ { a d v } ( t )$ is used to recompute $\hat { \pmb { x } } _ { a d v } ( t + d t )$ , retrieving the required dependencies. Then, we recursively obtain the gradient w.r.t. $\hat { \pmb x } _ { a d v } ( t )$ from the gradients w.r.t. $\hat { \pmb { x } } _ { a d v } ( t + d t )$ using this recovered sub-graph (see §3.2). Finally, gradients are backpropagated in a standard manner from $\pmb { x } _ { a d v } ( t ^ { * } )$ to $\scriptstyle A _ { \theta }$ to update $\pmb { x } _ { a d v }$ .

In §3.2 and §3.3, we revisit $D B P ^ { \circ } { \bf s }$ previous robustness, attributing it to flawed backpropagation and improper evaluation protocols. Most works [20, 30, 26, 40] judge attacks based on a single purification of the final adversarial input. While this seemingly mirrors $D B P ^ { \circ } { \bf s }$ intended “purify once and classify” deployment [30, 40], it is statistically invalid: since DBP samples noise randomly, one evaluation of the AE fails to capture true single misclassification probability across possible purifications. Moreover, due to $D B P ^ { \prime } { \bf s }$ memory-intensive gradients, prior works implement exact backpropagation [20, 26, 41] via checkpointing [8]. We discover and fix flaws in all previous implementations, introducing DiffGrad—the first reliable module for exact backpropagation through DBP (see Fig.1). In §4, we show that even under the flawed one-sample evaluation protocol, AutoAttack [10] significantly outperforms prior works, exposing their gradient fallacies. We further use a multi-sample evaluation that captures $D B P ^ { \circ } { \bf s }$ stochasticity and resubmission risk—aligned with DiffHammer [41], which also identified the evaluation gap, but retained flawed gradients. By fixing both, we prove DBP even more vulnerable than they report, with robustness dropping below 17.19%. Finally, we show prior attack enhancements tailored to DBP (e.g., DiffHammer, DiffAttack [20]), being oblivious to the ability we discover of standard gradient-based attacks to reshape its behavior, in fact disrupt optimization, lowering attack performance, and confirming our theoretical scrutiny.

To replace the current inherently vulnerable single-evaluation DBP defense scheme, we propose a more robust majority-vote (MV) setup, where multiple purified copies are classified and the final decision is based on the majority label. Yet, even under MV, DBP retains only partial robustness against norm-bounded attacks—see §4.2, supporting our theoretical findings. As we established the inefficacy of intermediate projections, this resistance is due to $D B P ^ { \circ } { \bf s }$ vast stochasticity: common AEs that introduce large changes to individual pixels are diluted and may not impact most paths. Instead, we require stealthy modifications that affect many pixels. In §5, we propose a novel adaptation of a recent strategy [21] from image watermarking that crafts low-frequency perturbations, accounting for links between neighboring pixels. This technique defeats DBP even under MV

Contributions. (i) Analytically scrutinizing adaptive attacks on DBP, proving they nullify theoretical claims regarding its robustness. (ii) Addressing protocol and gradient flaws in prior attacks, enabling reliable evaluations, and demonstrating degraded performance of DBP and the ineffectiveness of recent attack enhancement strategies [20, 41]. (iii) Introducing and evaluating a statistically grounded MV defense protocol for DBP. (iv) Proposing and adapting low-frequency (LF) attack optimizations to DBP, achieving unprecedented success even under MV. (v) Availability: aside from scalability and backpropagation issues, existing DBP implementations and attacks lack generalizability. We provide $D i f f B r e a k ^ { 1 } -$ the first toolkit for evaluating any classifier with DBP under various optimization methods, including our novel $L F ,$ using our reliable DiffGrad module for backpropagation. (vi) Extensive evaluations on ImageNet and CIFAR-10 prove our methods defeat $\bar { D B P }$ , bringing its robustness to $\sim 0 \%$ , outperforming previous works by a large margin.

## 2 Background & Related Work

Adversarial Attacks. Given x P $\mathbb { R } ^ { d }$ with true label $y ,$ classifier (model) $\mathcal { M } .$ , and preprocessing defense G (when no defense is used, $G \equiv I d )$ , attackers aim to generate a similar $\mathbf { \Delta } \mathbf { x } _ { a d v }$ s.t. $\mathcal { M } \big ( G ( \bar { \mathbf { x } _ { a d v } } ) \big ) \ne y .$ Crafting $\mathbf { { x } } _ { a d v }$ is formalized as:

$$
\pmb { x } _ { a d v } = \underset { \pmb { \mathcal { D } } ( \pmb { x } ^ { \prime } , \pmb { x } ) \leqslant \epsilon _ { \pmb { \mathcal { D } } } } { \arg \operatorname* { m i n } } \mathbb { E } [ \ell _ { G } ^ { \mathcal { M } } ( \pmb { x } ^ { \prime } , \ y ) ]
$$

for loss $\ell _ { G } ^ { \mathcal { M } }$ . Typically, $\ell _ { G } ^ { \mathcal { M } } ( \boldsymbol { \mathbf { \mathit { x } } } , \boldsymbol { \mathbf { \mathit { y } } } ) = \ell ( \mathcal { M } ( G ( \boldsymbol { \mathbf { \mathit { x } } } ) ) , \boldsymbol { \mathbf { \mathit { y } } } )$ , where ℓ is a loss over $\mathcal { M } \mathrm { { s } }$ output that captures the desired outcome. For instance, $\ell ( \mathcal { M } ( G ( \pmb { x } ) ) , y )$ can be chosen as the probability that the classifier’s output label is y, which we strive to minimize. D is a distance metric that ensures similarity if kept below some $\epsilon _ { x }$ . These untargeted attacks are the focus of many works [30, 40, 32, 29]. The expected value accounts for potential stochasticity in $G \left( { \bf e . g . , } D B P \right.$ below).

Diffusion models (DMs) [34, 36] learn to model a distribution p on $\mathbb { R } ^ { d }$ by reversing the process that diffuses inputs into noise. DMs involve two stochastic processes. The forward pass converts samples into pure Gaussians, and is governed by the following SDE for an infinitesimal step $d t > 0$

$$
\begin{array} { r } { d \pmb { x } = \pmb { f } ( \pmb { x } , t ) d t + g ( t ) d \pmb { w } . } \end{array}\tag{1}
$$

eq. (1) describes a stochastic integral whose solution up to $t ^ { * } \in [ 0 , 1 ]$ gives x $( t ^ { * } )$ . Here, $\pmb { f } : \mathbb { R } ^ { d } \times \mathbb { R } \longrightarrow$ $\mathbb { R } ^ { d }$ is the drift, $\boldsymbol { w } : \mathbb { R } \longrightarrow \mathbb { R } ^ { d }$ is a Wiener process, and $g : \mathbb { R } \longrightarrow \mathbb { R }$ is the diffusion coefficient. We focus on VP-SDE, which is the most common DM for $D B P \left[ 3 0 , 4 8 , 4 0 \right]$ . Yet, our insights generalizea to all DMs (see [36] for a review). In VP-SDE, $\begin{array} { r } { \pmb { f } ( \pmb { x } , \ t ) = - \frac { 1 } { 2 } \beta ( t ) \pmb { x } ( t ) } \end{array}$ and $g ( t ) = \sqrt { \beta ( t ) }$ , where $\beta ( t )$ is a noise scheduler outputting small positive constants. These choices yield a closed-form solution:

$$
\pmb { x } ( t ^ { * } ) = \sqrt { \alpha ( t ^ { * } ) } \pmb { x } + \sqrt { 1 - \alpha ( t ^ { * } ) } \pmb { \epsilon }\tag{2}
$$

for $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , I _ { d } )$ and $\alpha ( t ) = e ^ { - \int _ { 0 } ^ { t } \beta ( s ) d s }$ . With proper parameters, we have $\pmb { x } ( 1 ) \sim \mathcal { N } ( \mathbf { 0 } , \pmb { I } _ { d } )$ . Thus, a process that inverts eq. $( l )$ from $t ^ { * } = 1$ to 0 allows generating samples in p from random noise. Due to Anderson [1], the reverse pass is known to be a stochastic process with:

$$
d \hat { \pmb { x } } = [ \pmb { f } ( \hat { \pmb { x } } ( t ) , t ) - g ^ { 2 } ( t ) \nabla _ { \hat { \pmb { x } } ( t ) } \log p _ { t } ( \hat { \pmb { x } } ( t ) ) ] d t + g ( t ) d \bar { \pmb { w } } .\tag{3}
$$

Defining $\hat { \pmb x } ( t ^ { * } ) : = \pmb x ( t ^ { * } )$ , the process evolves from $t ^ { * }$ to 0 with a negative time step dt and reverse-time Wiener process $\bar { \pmb w } ( t )$ Let $p ( { \pmb x } )$ be the probability of x under $p ,$ and $p _ { 0 t } ( { \tilde { \mathbf { x } } } | { \tilde { \mathbf { x } } } )$ the conditional density atş t given $\mathbf { x } ( 0 ) ~ = ~ \mathbf { x }$ Then, the marginal density is given by $p _ { t } ( { \tilde { { \pmb x } } } ) \ =$ $\int p ( { \pmb x } ) p _ { 0 t } ( { \tilde { \pmb x } } | { \pmb x } ) d { \pmb x }$ , where $p _ { 0 } \equiv p .$ Solving eq. (3) requires the score $\ddot { \nabla } _ { \hat { \pmb { x } } ( t ) }$ q log $p _ { t } ( { \hat { \mathbf { x } } } ( t ) )$ , which can be approximated via a trained model sθ s.t. $\begin{array} { r } { \pmb { s } _ { \theta } ( \hat { \pmb { x } } ( t ) , t ) \approx \nabla _ { \hat { \pmb { x } } ( t ) } \log p _ { t } ( \hat { \pmb { x } } ( t ) ) } \end{array}$ at any point [36], yielding:

$$
d \hat { { \pmb x } } = - \frac { 1 } { 2 } \beta ( t ) [ \hat { { \pmb x } } ( t ) + 2 { \pmb s } _ { \theta } ( \hat { { \pmb x } } ( t ) , t ) ] d t + \sqrt { \beta ( t ) } d \bar { { \pmb w } } .\tag{4}
$$

As no closed-form solution exists, the process runs iteratively over discrete negative steps dt. Starting from $t ^ { * }$ , dxˆ is calculated at each $\begin{array} { r } { \dot { \boldsymbol { i } } = | \frac { \dot { \boldsymbol { t } } } { d t } | } \end{array}$ , until $t = 0 ,$ . This continuous-time DM describes a stochastic integral (despite the discretized implementations). An alternative, Denoising diffusion probabilistic modeling (DDPM) [19, 34], considers a discrete-time DM. Effectively, the two are equivalent [36].

DBP [30, 48, 40, 51] performs purification by diffusing each input x until optimal time $t ^ { * }$ that preserves class semantics while still diminishing malicious interruptions. $\pmb { x } ( t ^ { * } )$ is then given to the reverse pass, reconstructing a clean ${ \hat { \mathbf { x } } } ( 0 ) { \approx } { \pmb x } ~ \mathrm { s . t . } ~ { \hat { \mathbf { x } } } ( 0 ) { \sim } p$ for correct classification.

Certified vs. Empirical Robustness. While recent work has explored certified guarantees for diffusion models via randomized smoothing (RS) [48, 5, 55, 6, 45], these guarantees hold only under restrictive assumptions $- \mathrm { e . g . }$ , small $\ell _ { 2 }$ perturbations and thousands of Monte Carlo samples per input. In contrast, empirical defenses [40, 30, 31, 7, 53] aim for practical robustness under realistic threats [10] and efficient inference. Like prior work [41, 20, 30, 40, 24], we explicitly target these empirical defenses–—not orthogonal certified variants whose guarantees do not hold under stronger practical perturbations or operational constraints. Certification protocols are computationally prohibitive, and their theoretical bounds fail to capture threats that remain imperceptible but exceed certified radii.

## 3 Revisiting Diffusion-Based Purification

We study gradient-based attacks. In §3.1, we theoretically prove that adaptive attacks invalidate $D B P \mathrm { { ^ { s } _ { s } } }$ principles. Then, we reconcile this with previous findings of DBP’s robustness, attributing them to backpropagation flaws and improper evaluation protocols. In §3.2, we analyze backpropagation flaws in previous works and propose fixes. Finally, we present an improved evaluation protocol for DBP in §3.3 to better measure its robustness.

## 3.1 Why DBP Fails: Theoretical Vulnerability to Adaptive Attacks

$D B P ^ { \circ } { \bf s }$ robustness is primarily attributed to its ability to project the purified ${ \hat { \mathbf { x } } } ( 0 )$ onto the natural manifold [30, 48, 20], a property relied upon by both empirical and certified defenses. This foundational assumption—rooted in the inherent behavior of the purification process—has thus far remained unchallenged. In fact, the most advanced attacks have been explicitly tailored to exploit or circumvent it: DiffAttack [20] introduces suboptimal per-step losses to perturb the purification path (see §4), while DiffHammer [41] constrains its optimization to feasible paths that evade detection by the DBP process itself. DBP is often justified through its marginal consistency: since $\begin{array} { r } { \pmb { s } _ { \theta } \approx \nabla \log p _ { t } } \end{array}$ $\{ \pmb { x } ( t ) \} _ { t \in [ 0 , 1 ] }$ and $\{ \hat { \pmb x } ( t ) \} _ { t \in [ 0 , 1 ] }$ follow the same marginals [1], yielding ${ \hat { \mathbf { x } } } ( 0 ) \sim p .$ Specifically, Xiao et al. [48] show that:

$$
\begin{array} { r } { \operatorname* { P r } ( \hat { { \pmb x } } ( 0 ) | { \pmb x } ) \propto p ( \hat { { \pmb x } } ( 0 ) ) \cdot e ^ { \textstyle - \frac { \alpha ( t ^ { * } ) \| \hat { { \pmb x } } ( 0 ) - { \pmb x } \| _ { 2 } ^ { 2 } } { 2 ( 1 - \alpha ( t ^ { * } ) ) } } } \end{array}
$$

where p is the density of the natural data distribution and $\alpha ( t ^ { * } )$ is the variance schedule at time $t ^ { * }$ Thus, $D B P$ is expected to reject adversarial examples by construction: the probability of producing any ${ \hat { \pmb x } } ( 0 )$ that is both adversarial and lies off-manifold is exponentially suppressed by the score model. However, this reasoning assumes the reverse process remains faithful to $p .$ In practice, the score function sθ is itself an ML model—differentiable and susceptible to adversarial manipulation. Our key insight is that traditional gradient-based attacks, when backpropagated through DBP, do not simply bypass the purifier—they implicitly target it, steering the score model to generate samples from an adversarial distribution. This challenges the assumptions underlying prior attack strategies, many of which distort gradients or constrain optimization in ways that ignore the score model’s vulnerability, undermining their own effectiveness. Below, we formally characterize this vulnerability.

Definitions. Diffusion Process & Score Model. For $t _ { 1 } \geqslant t _ { 2 }$ , let $\hat { \pmb { x } } _ { t _ { 1 } : t _ { 2 } }$ represent the joint reverse diffusion trajectory ${ \hat { \pmb { x } } } ( t _ { 1 } ) , { \hat { \pmb { x } } } ( t _ { 1 } + d t ) , \ldots , { \hat { \pmb { x } } } ( t _ { 2 } )$ Let $\scriptstyle { \pmb { s } } \theta$ denote the score model used by $D B P$ to approximate the gradients of the natural data distribution at different time steps. $s _ { \theta ^ { t } }$ is the abstract score model invoked at time step t with parameters $\theta ^ { t }$ , corresponding to $s _ { \theta } ( \cdot , t )$ . Given that the score model $\scriptstyle { \pmb { s } } \theta$ is an ML model that interacts with adversarial input $^ { x , }$ we denote the parameters at each reverse step as $\theta _ { x } ^ { t } ,$ , capturing their dependence on x. That is, $\theta _ { \pmb { x } } ^ { t } \equiv \theta ^ { t } ( \pmb { x } )$ , which makes explicit that the purification process is not immutable, as adversarial modifications to x can shape its behavior.

Classifier. For a classifier M and label $y ,$ let $\mathcal { M } ^ { y } ( \pmb { u } )$ denote the probability that u belongs to class y. Adaptive Attack. A gradient-based algorithm that iteratively updates x with learning rate $\eta > 0 \colon$

$$
\mathbf { \Psi } \mathbf { x } = \mathbf { x } - \eta \widetilde { \nabla } _ { \mathbf { x } } , \quad \mathrm { w h e r e } \quad \widetilde { \nabla } _ { \mathbf { x } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \nabla _ { \mathbf { x } } [ \mathcal { M } ^ { y } ( \hat { \mathbf { x } } ( 0 ) _ { n } ) ] .
$$

Here, $\tilde { \nabla } _ { x }$ is the empirical gradient estimate over N purified samples ${ \hat { \pmb x } } ( 0 ) _ { n }$ , obtained via Monte Carlo approximation and backpropagated through $D B P ^ { \prime } { \bf s }$ stochastic process.

Theorem 3.1. The adaptive attack optimizes the entire reverse diffusion process, modifying the parameters $\{ \theta _ { x } ^ { t } \} _ { t \leqslant t ^ { * } }$ such that the output distribution $\hat { \pmb { x } } ( 0 ) \sim D B P ^ { \{ \theta _ { x } ^ { t } \} } ( { \pmb { x } } )$ where $D B P ^ { \{ \theta _ { x } ^ { t } \} } ( x )$ is the DBP pipeline with the score model’s weights $\{ \theta _ { x } ^ { t } \} _ { t \leqslant t ^ { * } }$ adversarially aligned. That is, the adversary implicitly controls the purification path and optimizes the weights $\{ \theta _ { { \boldsymbol { x } } } ^ { t } \} _ { t \leqslant t ^ { * } }$ to maximize:

$$
\operatorname* { m a x } _ { \{ \theta _ { x } ^ { t } \} _ { t \leqslant t } \ast } \mathbb { E } _ { \hat { \pmb { x } } ( 0 ) \sim D B P ^ { \{ \theta _ { x } ^ { t } \} } ( \pmb { x } ) } [ \mathrm { P r } ( \lnot y | \hat { \pmb { x } } ( 0 ) ) ] .
$$

Since $\{ \theta _ { x } ^ { t } \} _ { t \leqslant t ^ { * } }$ depend on the purification trajectory $\hat { \pmb { x } } _ { t * : 0 } ,$ , optimizing x under the adaptive attack directly shapes the purification process itself, forcing DBP to generate adversarially structured samples that maximize misclassification.

Impact. As DBP is widely perceived as immutable, its robustness is often attributed to gradient masking (vanishing or explosion) due to its presumed ability to project inputs onto the natural manifold. However, this reasoning treats DBP as a static pre-processing step rather than a dynamic optimization target. Our result overturns this assumption: rather than resisting adversarial influence, DBP itself becomes an active participant in adversarial generation. Crucially, standard adaptive attacks that backpropagate accurate gradients through $D B \bar { P }$ do not merely evade it—they implicitly exploit and reshape its behavior, forcing the purification process to align with the attacker’s objective. The introduced perturbations directly shape the purification trajectory, causing DBP to generate, rather than suppress, adversarial samples. Thus, DBP does not inherently neutralize adversarial perturbations but instead shifts the optimization target from the classifier to the score model $\begin{array} { r } { { \pmb { s } } \theta ; } \end{array}$ , leaving this paradigm highly vulnerable.

Proof. (Sketch) The adaptive attack maximizes the probability that the classifier mislabels the purified output, which is expressed as an expectation over the stochastic purification process. Applying the law of the unconscious statistician (LOTUS), we reformulate this expectation over the joint distribution of the reverse diffusion trajectory, shifting the dependency away from the classifier’s decision at the final step. Leveraging smoothness, we interchange differentiation and expectation, revealing that the adaptive attack’s gradient corresponds to the expected gradient of the purification trajectory, which is governed by the score model’s parameters. Crucially, these parameters are optimized implicitly through perturbations to x, rather than being explicitly modified. This demonstrates that the adaptive attack does not merely navigate DBP— it fundamentally shapes its behavior, exploiting the score model rather than directly targeting the classifier. The full proof is in Appendix C. □

## 3.2 Precise DBP Gradients with DiffGrad

Building on our theoretical analysis, we examine why prior adaptive attacks [30, 40, 20, 26, 24] underperform and show how DiffGrad, provided by our DiffBreak toolkit, resolves these limitations.

We reserve the terms forward/reverse pass for DBP and forward/backpropagation for gradient computations. Let $\hat { \pmb x } ( t ^ { * } ) : = \pmb x ( t ^ { * } )$ . The reverse process follows:

$$
\pmb { \hat { x } } ( t + d t ) = \pmb { \hat { x } } ( t ) + d \pmb { \hat { x } } ( t ) , \quad d t < 0 ,\tag{5}
$$

and by chain rule the gradient of any function F of ${ \hat { \mathbf { x } } } ( t + d t )$ w.r.t. xˆptq is:

$$
\nabla _ { \hat { \pmb { x } } ( t ) } F = \nabla _ { \hat { \pmb { x } } ( t ) } \langle \hat { \pmb { x } } ( t + d t ) , \nabla _ { \hat { \pmb { x } } ( t + d t ) } F \rangle .\tag{6}
$$

Applying this recursively yields gradients w.r.t. $\hat { \pmb x } ( t ^ { * } )$ , and via eq. (2), the input x. Yet, standard automatic differentiation is impractical for DBP due to excessive memory overhead from dependencies between all ${ \hat { \mathbf { x } } } ( t )$ . Prior work resorts to approximations such as the adjoint method [25], or checkpointing [8] to compute the exact gradients. However, approximations yield suboptimal attack performance [24]. On the other hand, we identify critical issues in existing checkpointing-based implementations that previously led to inflated estimates of $D B P ^ { \prime } { \bf s }$ robustness (see Appendix D for detailed analysis, the pseudo-code of our DiffGrad module, and empirical evaluations of backpropagation flaws):

1) High-Variance Gradients: Gradient-based attacks require estimating expected gradients using N Monte Carlo (EOT) samples. When N is small, gradient variance is high, leading to unreliable updates. Prior works used small N due to limitations in how EOT samples were purified—one at a time via serial loops in standard attack benchmarks (e.g., AutoAttack [10]). DiffGrad parallelizes EOT purifications. This integrates seamlessly with standard attacks. Coupled with our termination upon success protocols—see §4.2, this enables us to use up to 128 EOT samples (for CIFAR-10) with a speedup of up to 41ˆ upon early termination, drastically reducing variance.

2) Time Consistency: $t o r c h s d e ^ { 2 }$ is the de facto standard library for SDE solvers. Hence, several prior checkpointing approaches likely use torchsde as their backend. Yet, torchsde internally converts the integration interval into a PyTorch tensor, which causes a discrepancy in the time steps on which the score model is invoked during both propagation phases due to rounding issues if the checkpointing module is oblivious to this detail. DiffGrad ensures time steps match in both phases.

3) Reproducing Stochasticity: Forward and backward propagation must use identical randomness.   
We introduce a structured noise sampler, which stores and reuses noise realizations.

4) Guidance Gradients: Guided DBP [40] uses guidance metrics along the reverse pass. This guidance is obtained by applying a guidance function $\scriptstyle { \pmb { g } } _ { f n }$ to (potentially) both the original input x and the reverse-diffused xˆptq. As such, it creates paths from x to the loss that basic checkpointing fails to consider. Furthermore, the guidance itself is often in the form of a gradient, necessitating second-order differentiation during backpropagation, which is, by default, disabled by differentiation engines. We extend DiffGrad to include these “guidance" gradients. Critically, recent SOTA DBP defenses generate purified outputs from pure noise, relying on x only via guidance (see §6). As such, the lack of support for guidance gradients in all previous implementations leads to a false sense of security. To our knowledge, DiffGrad is the first to account for this component.

5) Proper Use of The Surrogate Method [24]: Upon inspecting DiffHammer’s code, we find their checkpointing implementation effectively applies Lee and Kim [24]’s surrogate approximation method to calculate the gradients—see Appendix G, which degrades gradient quality and impacts attack performance. Worse yet, DiffHammer’s implementation of the surrogate process suffers from severe flaws that further corrupt its gradients. We defer details to Appendix D.

## 3.3 On The Flaws in DBP’s Evaluation Protocols.

DBP is typically deployed in a single-purification (SP) setting: an input x is purified once via a stochastic reverse process into xˆp0q and classified as $c = \mathcal { M } ( \hat { \pmb x } ( 0 ) ) \ [ 3 0 , 4 0 ]$ . This assumes xˆp0q lies on the natural manifold and yields a correct label with high probability—see §3.1. However, prior evaluations [20, 30, 40] fail to assess this very property, introducing a critical mismatch: they evaluate robustness by purifying only the final adversarial iterate, testing just a single stochastic path, yielding noisy and unrepresentative estimates.

DiffHammer [41] addressed this with worst-case robustness (Wor.Rob): the attack succeeds if any of N purifications (evaluated at each attack step) leads to misclassification. The metric is Wor.Rob $\begin{array} { r } { \mathrel { \mathop : } = 1 - \frac { 1 } { S } \sum _ { j = 1 } ^ { S } \underset { i \in \| N \| } { m a x } \mathcal { A } _ { i } ^ { ( j ) } } \end{array}$ , where $A _ { i } ^ { ( j ) } = 1$ if the i-th purification of sample j fails, and $S$ is the dataset size. They justify this protocol by modeling resubmission attacks (e.g., in login attempts), thereby limiting N to (typically) $N = 1 0$ after which the attacker will be blocked. Yet attackers can retry inputs arbitrarily in stateless settings like spam, CSAM, and phishing. Even in stateful systems, the defender has no control over the stochastic path. Thus, SP evaluations require many purifications per input for statistically meaningful conclusions.

Yet, any stochastic defense fails given enough queries [27]. Even if DBP nearly always projects to the manifold, SP is only reliable if misclassification probability is negligible over all paths—rare in practice (§4). Robust predictions must hence aggregate over multiple purifications. We thus propose:

Majority Vote (MV). Given input x, generate K purifications and predict by majority: $c =$ $M V ^ { K } ( \pmb { x } ) : = m o d e \{ \mathcal { M } ( \pmb { x } ^ { ( 1 ) } ( 0 ) ) , \dots , \mathcal { M } ( \pmb { x } ^ { ( K ) } ( 0 ) ) \}$ Unlike certified smoothing [48], MV requires no excessive sampling, yet offers a stable, variance-tolerant robustness estimate, albeit without formal guarantees. We evaluate attacks using the majority-robustness (MV.Rob) metric:ř “ ‰ MV.Rob $\begin{array} { r } { : = 1 - \frac { 1 } { S } \sum _ { j = 1 } ^ { S } \mathbb { 1 } \left[ M V ^ { K } ( \pmb { x } _ { j } ) \neq y _ { j } \right] } \end{array}$ , where $y _ { j }$ is the ground-truth label. DiffHammer’s additional Avg.Rob $\begin{array} { r } { : = 1 - { \frac { 1 } { N S } } \sum _ { j = 1 } ^ { S } \sum _ { i = 1 } ^ { N } { \mathcal { A } } _ { i } ^ { ( j ) } } \end{array}$ averages path-level failures across all samples, conflating individual copy failures with actual sample robustness. A few fragile samples can dominate this metric, making it unreliable for deployment.

## 4 Experiments

We reevaluate DBP, demonstrating its degraded performance when evaluation and backpropagation issues are addressed, cementing our theoretical findings from §3.1.

Setup. We evaluate on CIFAR-10 [22] and ImageNet [11], using 256 random test samples for each consistent with previous work [41]. We consider two foundational DBP defenses: The VP-SDE DBP (DiffPure) [30] and the Guided-DDPM (see §3.2), GDMP [40]. We use the DMs [12, 19, 36] studied in the original works, adopting the same purification settings—See Appendix F. Following Nie et al. [30], we use WideResNet-28-10 and WideResNet-70-16 [52] for CIFAR-10, and WideResNet-50-2 and DeiT-S [14] for ImageNet. As in previous work [40, 30, 20, 41, 26, 24], we focus on the whitebox setting and use AutoAttack- $\mathcal { l } _ { \infty } \left( A \bar { A } - \ell _ { \infty } \right) \left[ 1 0 \right]$ with $\epsilon _ { \infty } = 8 / 2 5 5$ for CIFAR-10 and $\epsilon _ { \infty } = 4 / 2 5 5$ for

ImageNet. Existing works focus on norm-bounded $( \ell _ { \infty }$ and $\ell _ { 2 } )$ . For $D B P , \ell _ { \infty }$ has repeatedly proven superior [20, 24, 26], making it the focus of our evaluations.

## 4.1 Reassessing One-Shot DBP Robustness with Accurate Gradients

As noted in §3, $D B P \mathrm { ^ { * } s }$ robustness stems from two flaws: inaccurate gradients and flawed evaluation. As our work offers enhancements on both fronts, we evaluate each factor separately. Here, we isolate the gradient issue by re-running prior experiments under the same (flawed) single-evaluation protocol, but with accurate gradients via our DiffGrad module and compare the results to those from the literature. Despite using only 10 optimization steps (vs. up to 100 in prior work), our method significantly outperforms all gradient approximations and even recent full-gradient methods. For DiffPure, we lower robust accuracy from 62.11% (reported by Liu et al. [26]) to 48.05%, and for GDMP, from 24.53% (Surrogate [24]) to 19.53%. These results expose major flaws in existing gradient implementations and invalidate claimed gains from enhancement strategies like DiffAttack. Full breakdowns and additional baselines are provided in Appendix G.

Evaluations Under Liu et al. [26]’s Protocol. Liu et al. [26], like DiffHammer [41] and our own analysis, note that evaluating a single purification at attack termination inflates robustness scores. To address this, they propose a refined 1-evaluation protocol, which tests 20 purifications of the final AE and declares success if any fail. This offers a stricter assessment than earlier one-shot methods, though still weaker than Wor.Rob (see §3.3). Accordingly, Liu et al. [26] group their method with one-shot evaluations. Repeating their setup using DiffGrad (20 attack iterations, N “ 10 EOT), we observe a dramatic drop in robust accuracy: on WideResNet-28-10, we improve upon Liu et al. [26]’s results by 25.39% and 30.42% for DiffPure and GDMP, respectively—bringing the Rob Accs down to 30.86% and 10.55%. These results confirm the superiority of DiffGrad’s gradients and expose DBP’s realistic vulnerability. Detailed results are in Appendix H.

## 4.2 DBP Under Realistic Protocols

Here, we continue to highlight the flaws of one-shot evaluation and the strength of DiffGrad, while exposing the invalidity of previous attack enhancements over the standard gradient-based methodologies. To do so, we adopt a Wor.Rob protocol, similar to DiffHammer—see §3.3 and also include our majority-vote variant (MV.Rob). We reimplement DiffHammer and DiffAttack using DiffGrad, and compare them to their original versions from [41] and our standard $A A \mathrm { - } \ell _ { \infty }$ . All evaluations use CIFAR-10 with WideResNet-70-16; We also report ImageNet results below. Full denotes the standard AA attack that uses the full exact gradi-

Table 1: $A A \mathrm { - } \ell _ { \infty }$ performance comparison on CIFAR-10 $( \epsilon _ { \infty } { = } 8 / 2 5 5 )$ under realistic threat models. Metrics include Wor.Rob and MV.Rob under a 10-evaluation protocol. : indicates strategy is PGD.
<table><tr><td>Pur.</td><td>Gradient Method</td><td colspan="2">Wor.Rob % Cl-Acc Rob-Acc</td><td colspan="2">MV.Rob % Cl-Acc Rob-Acc</td></tr><tr><td rowspan="4">DiffPure [30]</td><td>BPDA DiffAttack (DiffHammer [41])</td><td rowspan="4">89.06</td><td>32.81 33.79</td><td rowspan="4">91.02</td><td>72.27 NA 39.45</td></tr><tr><td>DiffAttack-DiffGrad</td><td>15.63</td><td></td></tr><tr><td>DiffHammer (DiffHammer [41]) DiffHammer-DiffGrad</td><td>22.66 10.16</td><td>NA</td></tr><tr><td>Full (DiffHammer [41]) †</td><td>36.91</td><td>38.28 NA 39.45</td></tr><tr><td rowspan="6">GDMP [40]</td><td>BPDA</td><td rowspan="6">91.80</td><td>17.19 27.73</td><td rowspan="6">53.52 NA 14.45</td></tr><tr><td>DiffAttack (DiffHammer [41])</td><td>37.7</td></tr><tr><td>DiffAttack-DiffGrad</td><td>3.91</td></tr><tr><td></td><td>27.54</td></tr><tr><td>DiffHammer (DiffHammer [41])</td><td>7.81</td></tr><tr><td>DiffHammer-DiffGrad Full (DiffHammer [41]) † Full-DiffGrad</td><td>31.05</td></tr></table>

ents (i.e., via checkpointing). Ablation studies with larger evaluation sample counts for CIFAR-10 are in Appendix I.

Results. Table 1 reports Wor.Rob and MV.Rob scores. For fairness, we follow the exact setup from [41] using $N \stackrel { - } { = } 1 0$ samples both for EOT and evaluation copies. The attacks run for 100 iterations, terminating upon success (depending on the evaluated metric). For all attacks—Full, DiffAttack, and DiffHammer—DiffGrad yields significantly lower Wor.Rob compared to [41], confirming the gradient flaws discussed in §3.2 (MV.Rob is unique to our work). This establishes the need for our reliable DiffGrad as an essential tool for future progress in the field, given the repeated flaws that continue to surface in implementations of the checkpointing method. With correct gradients $( \mathrm { i . e . }$ , DiffGrad), all three attacks yield Wor.Rob ă 20%; for $G D M P , < 1 0 \%$ , exposing the failure of the single-purification defense and reinforcing the claim that a statistically resistant alternative (e.g.,

MV.Rob) must be used with DBP. Similarly, this highlights the flaws in the attack evaluation protocols that consider a single sample (i.e., 1-evaluation) as they drastically inflate robustness estimates.

Notably, minor differences in performance (among the three attacks) at this range where robustness has almost vanished and predictions are highly noisy reflect random variation, not real gains. Hence, we must focus on the MV.Rob when comparing the three, inspecting the results attained with our accurate DiffGrad gradient module. Under MV.Rob, DiffAttack and Full-DiffGrad are identical on DiffPure, and DiffHammer leads to MV.Rob lower by a mere 1.17%, which amounts to only 3 samples out of 256 and is thus statistically insignificant.

On GDMP, DiffHammer performs significantly worse than both others. Hence, we conclude that DiffHammer worsens attack performance in accordance with our theoretical findings—see§3.1. DiffAttack matches Full-DiffGrad on DiffPure; on GDMP, it shows a 2.35% edge, which despite the questionable statistical significance given the test set size, could indicate a potential advantage. However, our ImageNet comparisons below refute this hypothesis.

## ImageNet Evaluations.

We evaluate $A A \mathrm { - } \ell _ { \infty }$ on ImageNet using WideResNet-50-2 and DeiT-S classifiers under $\epsilon _ { \infty } { = } 4 / 2 5 5$ , following standard practice. For DeiT-${ \mathrm { \bf S } } ,$ we also reimplement DiffAttack via DiffGrad

Table 2: $A A { - } \ell _ { \infty }$ comparison on ImageNet $( \epsilon _ { \infty } = 4 / 2 5 5 )$
<table><tr><td rowspan="2">Models</td><td rowspan="2">Pur.</td><td rowspan="2">Gradient Method</td><td colspan="2">Wor.Rob %</td><td rowspan="2">MV.Rob % Cl-Acc Rob-Acc</td></tr><tr><td>Cl-Acc</td><td>Rob-Acc</td></tr><tr><td>WideResNet-50-2</td><td>Diff Pure [30]</td><td>Full-DiffGrad</td><td>74.22</td><td>12.11</td><td>77.02 29.69</td></tr><tr><td rowspan="3">DeiT-S</td><td>DiffPure [30]</td><td>DiffAttack-DiffGrad FullifGrad</td><td>73.63</td><td>25 21.09</td><td>77.34 42.21</td></tr><tr><td>GDMP [40]</td><td>Full-DiffGrad</td><td></td><td></td><td>32.81 32.83</td></tr><tr><td></td><td></td><td>69.14</td><td>20.70</td><td>75.0</td></tr></table>

for comparison. Each attack uses 16 EOT samples and 8 samples for prediction (Wor.Rob and MV.Rob). Attacks run for 100 iterations with early stopping. As with CIFAR-10, $D B P ` { \boldsymbol { \varsigma } }$ robustness drops sharply: Wor.Rob ranges from just 12.11% to 21.09%, while MV.Rob peaks at 32.83%. This confirms the vulnerability of single-purification and the strength of gradient-based attacks. DiffAttack underperforms our standard attack by 9.4% MV.Rob on DeiT-S, reinforcing its inferiority.

## 5 Defeating Increased Stochasticity

DBP’s stochasticity boosts its robustness under MV (see §4.2). Typical adversarial strategies incur high-frequency changes as they directly operate on pixels, altering each significantly w.r.t. its neighbors. This leads to visual inconsistencies, limiting the distortion budget. Such modifications are also easily masked by $D B P ^ { \circ } { \bf s }$ noise. Instead, systemic, low-frequency (LF) changes allow larger perturbations and resist randomness.

Our LF method is inspired by a recent attack— UnMarker [21]— on image watermarking that employs novel units termed Optimizable Filters (OFs). In signal processing, a filter is a smoothing operation defined by a kernel $\bar { \kappa } \in \mathbb { R } _ { + } ^ { M \times N }$ (with values that sum to 1), with which the input is convolved. The output at each pixel is a weighted average of all pixels in its $M { \times } N$ vicinity, depending on the weights assigned by K. Hence, filters incur systemic changes. Yet, they apply the same operation universally, unable to produce stealthy AEs, as the changes required to alter the label will be uniformly destructive. $O F s$ allow each pixel $( i , j )$ to have its own kernel $\kappa ^ { i , j }$ to customize the filtering at each point. $\kappa ^ { * }$ is the set of all per-pixel $\kappa ^ { i , j } { \bf s }$ . The weights $\theta _ { \kappa ^ { * } }$ are learned via feedback from a perceptual metric (lpips) [56], leading to an assignment that ensures similarity while maximizing the destruction at visually non-critical regions to optimize a specific objective. Note that the lpips constraint replaces the traditional norm constraint. To guarantee similarity, they also impose geometric constraints via color kernels $\sigma _ { c }$ , similar to guided filters (details in Appendix E.1).

We subject x to a chain $\prod _ { O F } ^ { B } \equiv O F _ { \kappa \ast _ { 1 } , { \bf { x } } , \sigma _ { c _ { 1 } } } \circ \cdot \cdot \cdot \circ O F _ { \kappa _ { B } ^ { \ast } , { \bf { x } } , \sigma _ { c _ { B } } }$ of OFs similar to UnMarker, replacing the objective pertaining to watermark removal in the filters’ weights’ learning process with the loss over M. Each OF has a kernel set $\kappa _ { b } ^ { * }$ K\* (with wights $\theta _ { \kappa _ { \mathrm { b } } ^ { * } }$ and shape $M _ { b } { \times } N _ { b } )$ , and $\sigma _ { c _ { b } } .$ . We optimize:

$$
\pmb { x } _ { a d v } = \pmb { a r g m i n } \left[ \begin{array} { c } { \ell _ { G } ^ { M } \big ( \frac { \mathtt { B } } { \Omega _ { F } } ( \pmb { x } + \pmb { \delta } ) , \ y \big ) } \\ { + c \cdot m a x \{ l p i p s ( \pmb { x } , \ \frac { \mathtt { B } } { \Omega _ { F } } ( \pmb { x } + \pmb { \delta } ) ) - \tau _ { p } , 0 \} } \end{array} \right]\tag{7}
$$

$\ell _ { G } ^ { \mathcal { M } }$ denotes any loss as defined in §2. δ is a modifier that directly optimizes x, similar to traditional attacks. AEs are generated by manipulating x via δ and propagating the result through the filters. While direct modifications alone do not cause systemic changes, with OFs, they are smoothed over neighbors of the receiving pixels. δ allows disruptions beyond interpolations. Similar to UnMarker, we chain several OFs with different shapes to explore various interpolations. Optimization is iterative (code in Appendix E.3). max $\{ l p i p s ( \pmb { x } , \prod _ { O F } ^ { B } ( \pmb { x } + \pmb { \delta } ) ) - \tau _ { p } , 0 \}$ enforces similarity: If the distance exceeds $\tau _ { p } ,$ the lpips gradients lower it in the next iteration. Otherwise, it returns 0, minimizing $\ell _ { G } ^ { \mathcal { M } }$ unconditionally. This gives a solution within the $\tau _ { p }$ constraint (violating outputs are discarded), yielding optimal $\{ \widehat { \theta } _ { \kappa \ast } \} , \widehat { \delta } \ \mathrm { s . t . } \ x _ { a d v } = \prod _ { O F } ^ { B } \bigl ( \pmb { x } + \widehat { \delta } \bigr )$ , where $\prod _ { O F } ^ { B }$ are the filters with $\{ \widehat \theta _ { \kappa \frac { * } { b } } \}$

## 5.1 DBP Against Low-Frequency AEs

Our final question is: Can DBP be degraded further under MV? Based on §5, this is possible with LF, which we test in this section. For LF (see §5), we use VGG-LPIPS [56] as the distance metric with $\tau _ { p } ~ = ~ 0 . 0 5$ , ensuring imperceptibility [17, 21]. Remaining parameters are similar to UnMarker’s (see Appendix E.2). We use 128 EOT samples for CIFAR-10 (same for label predictions since increasing the sample set size leads to enhanced

Table 3: Performance of LF attack under MV.
<table><tr><td>Pur.</td><td>Dataset</td><td>Models</td><td>Cl-Acc %</td><td>Rob-Acc %</td></tr><tr><td rowspan="5">DiffPure [30]</td><td rowspan="2">ImageNet</td><td>ResNet-50</td><td>72.54</td><td>0.00</td></tr><tr><td>WideResNet-50-2</td><td>77.02</td><td>0.00</td></tr><tr><td></td><td>DeiT-S</td><td>77.34</td><td>0.00</td></tr><tr><td rowspan="2">CIFAR-10</td><td>WideResNet-28-10</td><td>92.19</td><td>2.73</td></tr><tr><td>WideResNet-70-16</td><td>92.19</td><td>3.13</td></tr><tr><td rowspan="5">GDMP [40]</td><td rowspan="3">ImageNet</td><td>ResNet-50</td><td>73.05</td><td>0.39</td></tr><tr><td>WideResNet-50-2</td><td>71.88</td><td>0.00</td></tr><tr><td>De-s</td><td>75.00</td><td>0.39</td></tr><tr><td rowspan="2">CIFAR-10</td><td>WideResNet-28-10</td><td>93.36</td><td>0.00</td></tr><tr><td>WideResNet-70-16</td><td>92.19</td><td>0.39</td></tr></table>

robustness of DBP—see Appendix I). For ImageNet, the numbers are identical to §4.2 (note that the dimensionality of ImageNet makes larger sample sets prohibitive and our selected set sizes reflect realistic deployments— the largest number of samples that can fit into a modern GPU simultaneously).

The results for LF (using our Full-DiffGrad for backpropagation) are in Table 3. We also include a ResNet-50 classifier for ImageNet. $L F { ^ { \circ } } { _ { \mathrm { { S } } } }$ success is unprecedented: not only does it defeat all classifiers completely, leaving the strongest with Rob-Acc of 3.13%, but it also does so in the challenging MV setting, where previous approaches fail.

Concluding Remarks. Our findings highlight a limitation in current robustness evaluations, which focus heavily on norm-bounded attacks while overlooking powerful alternatives like low-frequency (LF) perturbations. Though not norm-bounded, LF is perceptually constrained—similar to StAdv [47], a longstanding benchmark against DBP Nie et al. [30]. $L F { ^ { \circ } } { _ { \mathbf { S } } }$ imperceptibility is guaranteed due to using a perceptual threshold $\tau _ { p } = 0 . 0 5$ that has previously been proven to guarantee stealthiness and quality [17, 21]. We include qualitative results in Appendix J confirming this. One may question if existing attacks like StAdv can also defeat $D B P$ under MV, rendering LF incremental. We address this in Appendix J, showing StAdv fails to produce stealthy AEs in this setting. While other techniques may also be adaptable to attacking certain DBP variants in the future, $L F$ enjoys a solid theoretical foundation such alternatives may lack, limiting their generalizability to all $\bar { D B P }$ defenses.

## 6 Potential Countermeasures

Adversarial Training (AT). While AT remains a leading defense, it performs poorly on unseen threats. As expected, it fails against our LF. On an adversarially trained WideResNet-28-10 for CIFAR-10 (DiffPure), LF reduces robust accuracy to just 0.78%, confirming this limitation.

SOTA DBP: MimicDiffusion. One might ask whether recent variants offer improved robustness. Most build incrementally on the foundational defenses in §4, meaning our results broadly generalize. One notable exception is MimicDiffusion [35], which generates outputs entirely from noise, using the input only as guidance (see §3.2). Its goal is to preserve semantic content for classification while eliminating adversarial perturbations, and it reports SOTA robustness to adaptive attacks.

However, based on our theoretical analysis, we hypothesize this robustness is illusory—an artifact of flawed evaluation and broken gradient flow. Unlike GDMP, which involves both direct and guidance paths from x to the loss, MimicDiffusion relies solely on guidance. Since guidance gradients require second-order derivatives (disabled by default), the original paper fails to compute meaningful gradients, severely overestimating robustness. We correct this using DiffGrad, the first method to properly differentiate through guidance paths. On CIFAR-10, using $A A \mathrm { - } \ell _ { \infty }$ (we exclude LF for time limitations) with N “ 128 for both EOT and evaluation, we find that the original Rob-Acc scores of 92.67% (WRN-28-10) and 92.26% (WRN-70-16) collapse under proper evaluation: Wor.Rob drops to 2.73% and 4.3%, respectively. Even under stricter MV.Rob, MimicDiffusion performs worse than the standard DBP defenses from §4, with accuracies of just 25.78% and 27.73%. This confirms that MimicDiffusion’s robustness is not real, and collapses under proper gradients.

These findings highlight the urgent need for defenses that account for more robust defenses.

## Acknowledgements

This work was supported by the NSERC Discovery Grant RGPIN-2020-04722 and the Waterloo-Huawei Joint Innovation Laboratory.

## 7 Conclusion

We scrutinized DBP’s theoretical foundations, overturning its core assumptions. Our analysis of prior findings revealed their reliance on flawed gradients, which we corrected to enable accurate evaluations, exposing degraded performance under adaptive attacks. Finally, we evaluated DBP in a stricter setup, wherein we found its increased stochasticity leaves it partially immune to norm-bounded AEs. Yet, our novel low-frequency approach defeats this defense in both settings. We find current DBP is not a viable response to AEs, highlighting the need for improvements.

## References

[1] Brian DO Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 1982.

[2] Anish Athalye, Nicholas Carlini, and David Wagner. Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. In International Conference on Machine Learning (ICML), 2018.

[3] Jacob Buckman, Aurko Roy, Colin Raffel, and Ian J. Goodfellow. Thermometer encoding: One hot way to resist adversarial examples. In International Conference on Learning Representations (ICLR), 2018.

[4] Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In IEEE Symposium on Security and Privacy (SP), 2017.

[5] Nicholas Carlini, Florian Tramer, Krishnamurthy Dj Dvijotham, Leslie Rice, Mingjie Sun, and J Zico Kolter. (Certified!!) adversarial robustness for free! In International Conference on Learning Representations (ICLR), 2023.

[6] Huanran Chen, Yinpeng Dong, Shitong Shao, Zhongkai Hao, Xiao Yang, Hang Su, and Jun Zhu. Your diffusion model is secretly a certifiably robust classifier. In International Conference on Machine Learning (ICML), 2024.

[7] Huanran Chen, Yinpeng Dong, Zhengyi Wang, Xiao Yang, Chengqi Duan, Hang Su, and Jun Zhu. Robust classification via a single diffusion model. In International Conference on Machine Learning (ICML), 2024.

[8] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174, 2016.

[9] Jeremy Cohen, Elan Rosenfeld, and J. Zico Kolter. Certified adversarial robustness via randomized smoothing. In International Conference on Machine Learning (ICML), 2019.

[10] Francesco Croce and Matthias Hein. Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. In International Conference on Machine Learning (ICML), 2020.

[11] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2009.

[12] Prafulla Dhariwal and Alexander Quinn Nichol. Diffusion models beat GANs on image synthesis. In Conference on Neural Information Processing Systems (NeurIPS), 2021.

[13] Guneet S. Dhillon, Kamyar Azizzadenesheli, Zachary C. Lipton, Jeremy Bernstein, Jean Kossaifi, Aran Khanna, and Animashree Anandkumar. Stochastic activation pruning for robust adversarial defense. In International Conference on Learning Representations (ICLR), 2018.

[14] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations (ICLR), 2021.

[15] Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. In International Conference on Learning Representations (ICLR), 2015.

[16] Chuan Guo, Mayank Rana, Moustapha Cissé, and Laurens van der Maaten. Countering adversarial images using input transformations. In International Conference on Learning Representations (ICLR), 2018.

[17] Qingying Hao, Licheng Luo, Steve TK Jan, and Gang Wang. It’s not what it looks like: Manipulating perceptual hashing based applications. In ACM SIGSAC Conference on Computer and Communications Security (CCS), 2021.

[18] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[19] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Conference on Neural Information Processing Systems (NeurIPS), 2020.

[20] Mintong Kang, Dawn Song, and Bo Li. DiffAttack: Evasion attacks against diffusion-based adversarial purification. In Conference on Neural Information Processing Systems (NeuIPS), 2024.

[21] Andre Kassis and Urs Hengartner. UnMarker: A Universal Attack on Defensive Image Watermarking . In IEEE Symposium on Security and Privacy (SP), 2025.

[22] Alex Krizhevsky. Learning multiple layers of features from tiny images, 2009.

[23] Alexey Kurakin, Ian J. Goodfellow, and Samy Bengio. Adversarial examples in the physical world. In Artificial intelligence safety and security. Chapman and Hall/CRC, 2018.

[24] Minjong Lee and Dongwoo Kim. Robust evaluation of diffusion-based adversarial purification. In IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[25] Xuechen Li, Ting-Kam Leonard Wong, Ricky TQ Chen, and David Duvenaud. Scalable gradients for stochastic differential equations. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2020.

[26] Yiming Liu, Kezhao Liu, Yao Xiao, ZiYi Dong, Xiaogang Xu, Pengxu Wei, and Liang Lin. Towards understanding the robustness of diffusion-based purification: A stochastic perspective. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=shqjOIK3SA.

[27] Keane Lucas, Matthew Jagielski, Florian Tramèr, Lujo Bauer, and Nicholas Carlini. Randomness in ML defenses helps persistent attackers and hinders evaluators. CoRR, 2023.

[28] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations (ICLR), 2018.

[29] Dongyu Meng and Hao Chen. MagNet: a two-pronged defense against adversarial examples. In ACM SIGSAC Conference on Computer and Communications Security (CCS), 2017.

[30] Weili Nie, Brandon Guo, Yujia Huang, Chaowei Xiao, Arash Vahdat, and Animashree Anandkumar. Diffusion models for adversarial purification. In International Conference on Machine Learning (ICML), 2022.

[31] Yidong Ouyang, Liyan Xie, and Guang Cheng. Improving adversarial robustness through the contrastiveguided diffusion process. In International Conference on Machine Learning (ICML), 2023.

[32] Tianyu Pang, Kun Xu, and Jun Zhu. Mixup inference: Better exploiting mixup to defend adversarial attacks. In International Conference on Learning Representations (ICLR), 2019.

[33] Jonas Rauber, Wieland Brendel, and Matthias Bethge. Foolbox: A python toolbox to benchmark the robustness of machine learning models. In Reliable Machine Learning in the Wild Workshop, 34th International Conference on Machine Learning, 2017. URL http://arxiv.org/abs/1707.04131.

[34] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning (ICML), 2015.

[35] Kaiyu Song, Hanjiang Lai, Yan Pan, and Jian Yin. MimicDiffusion: Purifying adversarial perturbation via mimicking clean diffusion model. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[36] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations (ICLR), 2021.

[37] Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian J. Goodfellow, and Rob Fergus. Intriguing properties of neural networks. In International Conference on Learning Representations (ICLR), 2014.

[38] Florian Tramèr, Nicholas Carlini, Wieland Brendel, and Aleksander Madry. On adaptive attacks to adversarial example defenses. In Conference on Neural Information Processing Systems (NeurIPS), 2020.

[39] Arash Vahdat, Karsten Kreis, and Jan Kautz. Score-based generative modeling in latent space. In Conference on Neural Information Processing Systems (NeurIPS), 2021.

[40] Jinyi Wang, Zhaoyang Lyu, Dahua Lin, Bo Dai, and Hongfei Fu. Guided diffusion model for adversarial purification. arXiv preprint arXiv:2205.14969, 2022.

[41] Kaibo Wang, Xiaowen Fu, Yuxuan Han, and Yang Xiang. DiffHammer: Rethinking the robustness of diffusion-based adversarial purification. In Conference on Neural Information Processing Systems (NeurIPS), 2024.

[42] Zekai Wang, Tianyu Pang, Chao Du, Min Lin, Weiwei Liu, and Shuicheng Yan. Better diffusion models further improve adversarial training. In International Conference on Machine Learning (ICML), 2023.

[43] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 2004.

[44] Eric Wong, Leslie Rice, and J. Zico Kolter. Fast is better than free: Revisiting adversarial training. In International Conference on Learning Representations (ICLR), 2020.

[45] Quanlin Wu, Hang Ye, and Yuntian Gu. Guided diffusion model for adversarial purification from random noise. arXiv preprint arXiv:2206.10875, 2022.

[46] Chang Xiao, Peilin Zhong, and Changxi Zheng. Enhancing adversarial defense by k-winners-take-all. In International Conference on Learning Representations (ICLR), 2020.

[47] Chaowei Xiao, Jun-Yan Zhu, Bo Li, Warren He, Mingyan Liu, and Dawn Song. Spatially transformed adversarial examples. In International Conference on Learning Representations (ICLR), 2018.

[48] Chaowei Xiao, Zhongzhu Chen, Kun Jin, Jiongxiao Wang, Weili Nie, Mingyan Liu, Anima Anandkumar, Bo Li, and Dawn Song. DensePure: Understanding diffusion models towards adversarial robustness. In Workshop on Trustworthy and Socially Responsible Machine Learning, NeurIPS, 2022.

[49] Cihang Xie, Jianyu Wang, Zhishuai Zhang, Zhou Ren, and Alan Yuille. Mitigating adversarial effects through randomization. In International Conference on Learning Representations (ICLR), 2018.

[50] Yuzhe Yang, Guo Zhang, Dina Katabi, and Zhi Xu. ME-Net: Towards effective adversarial robustness with matrix estimation. In International Conference on Machine Learning (ICML), 2019.

[51] Jongmin Yoon, Sung Ju Hwang, and Juho Lee. Adversarial purification with score-based generative models. In International Conference on Machine Learning (ICML), 2021.

[52] Sergey Zagoruyko. Wide residual networks. arXiv preprint arXiv:1605.07146, 2016.

[53] Boya Zhang, Weijian Luo, and Zhihua Zhang. Purify++: Improving diffusion-purification with advanced diffusion models and control of randomness. arXiv preprint arXiv:2310.18762, 2023.

[54] Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric P. Xing, Laurent El Ghaoui, and Michael I. Jordan. Theoretically principled trade-off between robustness and accuracy. In International Conference on Machine Learning (ICML), 2019.

[55] Jiawei Zhang, Zhongzhu Chen, Huan Zhang, Chaowei Xiao, and Bo Li. DiffSmooth: Certifiably robust learning via diffusion models and local smoothing. In USENIX Security Symposium (USENIX Security), 2023.

[56] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

## A Broader Impact & Limitations

This work advances adversarial robustness by exposing a fundamental vulnerability in Diffusion-Based Purification (DBP). We show that adaptive attacks do not merely circumvent DBP but repurpose it as an adversarial generator, invalidating its theoretical guarantees. While this insight highlights a critical security risk, it also provides a foundation for designing more resilient purification strategies. To facilitate rigorous and reproducible evaluation, we introduce $D i f f B r e a k .$ , an open-source toolkit providing the first reliable gradient module for DBP-based defenses, in addition to a wide array of attack implementations, including our low-frequency (LF) strategy that outperforms existing approaches, seamlessly applicable to a broad range of classifiers. Similar to existing adversarial attack toolkits (e.g., Foolbox [33], AutoAttack [10]), our framework is intended for research and defensive purposes. While this tool enhances robustness assessments, we acknowledge its potential misuse in adversarial applications. To mitigate risks, we advocate its responsible use for research and defensive purposes only. Finally, we urge developers of ML security systems to integrate our findings to design more resilient defenses against adaptive attacks. We encourage future research to explore fundamentally secure purification methods that are inherently resistant to manipulations.

Limitations. Our work focuses on dissecting the robustness of DBP defenses, both theoretically and empirically, with an emphasis on attack strategy. While we briefly consider existing mitigation techniques, future work should explore how diffusion models can be more effectively leveraged for defense. Classifiers could also be adversarially trained on LF perturbations to improve robustness. Yet, the vast hyperparameter and architecture space of LF potentially creates an effectively infinite threat surface, limiting the practicality of adversarial training. We leave this to future investigation.

While we do not report formal error bars or confidence intervals, this decision aligns with standard practice in the literature (e.g., [20, 26]) due to the high computational cost of gradient-based DBP attacks. Instead, we perform extensive experiments and also provide repeated empirical trials and variance analyses in Appendix D.2 that confirm our conclusions are robust, despite the absence of formal statistical intervals.

Although our main text focuses on conceptual and empirical flaws in DBP, Appendix D.1.3 and Appendix D.3 provide theoretical and asymptotic cost analysis, and Appendix I presents latency measurements and real-world speedup comparisons. Our module is efficient, scales linearly with diffusion steps, and outperforms prior implementations in both latency and throughput. Thus, while cost is not emphasized in the main paper, it is extensively addressed and poses no limitation in practice.

## B Equivalence of DDPM and VP-SDE

An alternative to the continuous-time view described in §2 (i.e., VP-SDE) is Denoising diffusion probabilistic modeling (DDPM) [19, 34], which considers a discrete-time framework (DM) where the forward and reverse passes are characterized by a maximum number of steps T . Here, the forward pass is a Markov chain:

$$
\pmb { x } _ { i } = \sqrt { 1 - \beta _ { i } } \pmb { x } _ { i - 1 } + \sqrt { \beta _ { i } } \pmb { z } _ { i }
$$

where $\smash { z _ { i } \sim \mathcal { N } ( \mathbf { 0 } , \ I _ { d } ) }$ and $\beta _ { i }$ is a small positive noise scheduling constant. Defining $\begin{array} { r } { d t = \frac { 1 } { T } } \end{array}$ , we know due to Song et al. [36] that when $T \longrightarrow \infty ( { \mathrm { i . e , ~ } } d t \longrightarrow 0$ , which is the effective case of interest), this converges to the SDE in eq. (1) (with the drift and diffusion function f and g described in §2). The reverse pass is also a Markov chain given as:

$$
d \hat { x } = \hat { x } _ { i - 1 } - \hat { x } _ { i } = \frac { 1 } { \sqrt { 1 - \beta _ { i } } } ( ( 1 - \sqrt { 1 - \beta _ { i } } ) \hat { x } _ { i } + \beta _ { i } s _ { \theta } ( \hat { x } _ { i } , i ) ) + \sqrt { \beta _ { i } } z _ { i }\tag{8}
$$

When $T \longrightarrow \infty , d { \hat { x } }$ converges to eq. (4) (see [36]). Thus, the two views are effectively equivalent. Accordingly, we focus on the continuous view, which encompasses both frameworks. For discrete time, xptq will be used to denote $\scriptstyle { \pmb { x } } _ { \mid { \frac { t } { d t } } \mid }$

## C Proof of Theorem 3.1

Theorem 3.1. The adaptive attack optimizes the entire reverse diffusion process, modifying the parameters $\{ \theta _ { x } ^ { t } \} _ { t \leqslant t ^ { * } }$ such that the output distribution $\hat { \pmb x } ( 0 ) \sim D B P ^ { \{ \theta _ { x } ^ { t } \} } ( { \pmb x } )$ , where $D B P ^ { \{ \theta _ { x } ^ { t } \} } ( x )$ is the

DBP pipeline with the score model’s weights $\{ \theta _ { x } ^ { t } \} _ { t \leqslant t ^ { * } }$ adversarially aligned. That is, the adversary implicitly controls the purification path and optimizes the weights $\{ \theta _ { { \boldsymbol x } } ^ { \bar { t } } \} _ { t \leqslant t ^ { * } }$ to maximize:

$$
\operatorname* { m a x } _ { \{ \theta _ { \alpha } ^ { t } \} } \mathbb { E } _ { \hat { \pmb { x } } ( 0 ) \sim D B P ^ { \{ \theta _ { \alpha } ^ { t } \} } ( \pmb { x } ) } [ \operatorname* { P r } ( \mathbf { \mathcal { - } } y | \hat { \pmb { x } } ( 0 ) ) ] .
$$

Proof. The adversary aims to maximize:

$$
\mathcal { Q } ( \pmb { x } ) \equiv \mathbb { E } _ { \hat { \pmb { x } } ( 0 ) \sim D B P ^ { \{ \theta ^ { t } \} } ( \pmb { x } ) } [ \mathrm { P r } ( \ - y | \hat { \pmb { x } } ( 0 ) ) ]
$$

Expanding this expectation yields the following alternative representations:

$$
\begin{array} { l } { \displaystyle \mathcal { Q } ( \pmb { x } ) = \int \operatorname* { P r } ( \mathrm {  } y | \hat { \pmb { x } } ( 0 ) ) p ( \hat { \pmb { x } } ( 0 ) ) d \hat { \pmb { x } } ( 0 ) } \\ { \displaystyle = \int \operatorname* { P r } ( \mathrm {  } y | \hat { \pmb { x } } ( 0 ) ) \int p ( \hat { \pmb { x } } ( 0 ) | \hat { \pmb { x } } _ { t ^ { \ast } : - d t } ) p ( \hat { \pmb { x } } _ { t ^ { \ast } : - d t } ) d \hat { \pmb { x } } _ { t ^ { \ast } : - d t } d \hat { \pmb { x } } ( 0 ) } \\ { \displaystyle = \int \operatorname* { P r } ( \mathrm {  } y | \hat { \pmb { x } } ( 0 ) ) p ( \hat { \pmb { x } } _ { t ^ { \ast } : 0 } ) d \hat { \pmb { x } } _ { t ^ { \ast } : 0 } . } \end{array}
$$

The first transition is due to the definition of expectation, the second follows from the definition of marginal probability, and the final transition replaces the joint density function with its compact form.

Since this formulation abstracts away x and $\{ \theta _ { x } ^ { t } \} _ { t \leqslant t ^ { * } }$ , we explicitly include them as $p ( \hat { \pmb x } _ { t ^ { * } : 0 } ) \equiv$ $p \big ( \hat { \pmb { x } } _ { t ^ { * } : 0 } | \pmb { x } , \{ \theta _ { \pmb { x } } ^ { t } \} _ { t \leqslant t ^ { * } } \big )$ For notation simplicity, we define $p _ { \theta _ { x } } ( \hat { \pmb x } _ { t } \ast _ { : 0 } | { \pmb x } ) \equiv p ( \hat { \pmb x } _ { t } \ast _ { : 0 } | { \pmb x } , \{ \theta _ { x } ^ { t } \} _ { t \leqslant t ^ { * } } )$ Thus, the final optimization objective is:

$$
\mathcal { Q } ( { \pmb x } ) = \int \mathrm { P r } ( \lnot y | \hat { \pmb x } ( 0 ) ) p _ { \theta _ { x } } ( \hat { \pmb x } _ { t ^ { * } : 0 } | { \pmb x } ) d \hat { \pmb x } _ { t ^ { * } : 0 } .
$$

Since we assume smoothness $( \mathcal { C } ^ { 2 } )$ , we interchange gradient and integral:

$$
\nabla _ { \pmb { x } } [ \mathcal { Q } ( \pmb { x } ) ] = \int \operatorname* { P r } ( - y | \hat { \pmb { x } } ( 0 ) ) \nabla _ { \pmb { x } } p _ { \theta _ { \pmb { x } } } ( \hat { \pmb { x } } _ { t ^ { * } : 0 } | \pmb { x } ) d \hat { \pmb { x } } _ { t ^ { * } : 0 } .
$$

where the last step is because $\operatorname* { P r } ( \lnot y | \hat { \pmb x } ( 0 ) )$ does not depend on x, but only on ${ \hat { \mathbf { x } } } ( 0 )$ which is independent of x (though its probability of being the final output of $D B P$ is a function of x). Optimizing the objective via gradient ascent relies on altering $D B P ^ { \prime } { \bf s }$ output distribution alone, evident in its gradient where $\bar { \mathrm { P r } } ( \lnot y | \hat { \pmb { x } } ( 0 ) )$ q assigned by the classifier is a constant with its gradient ignored. While we lack direct access to the gradients of the probabilistic paths above, we may still attempt to solve the optimization problem.

Recall that by definition, @u $\mathcal { M } ^ { y } ( \pmb { u } ) = \mathrm { P r } ( y | \pmb { u } )$ . Thus, the required gradient can also be expressed as:

$$
\nabla _ { \pmb { x } } \big [ \mathcal { Q } ( \pmb { x } ) \big ] = \nabla _ { \pmb { x } } \bigg [ \int \operatorname* { P r } (  y | \hat { \pmb { x } } ( 0 ) ) p _ { \theta _ { \pmb { x } } } ( \hat { \pmb { x } } _ { t ^ { * } : 0 } | \pmb { x } ) d \hat { \pmb { x } } _ { t ^ { * } : 0 } \bigg ] =
$$

$$
\nabla _ { \pmb { x } } \bigg [ \underset { p _ { \theta _ { \pmb { x } } } ( \hat { \pmb x } _ { t } * _ { 0 } | \pmb x ) } { \mathbb { E } } [ 1 - \mathcal { M } ^ { y } ( \hat { \pmb x } ( 0 ) ) ] \bigg ] = - \nabla _ { \pmb { x } } \bigg [ \underset { p _ { \theta _ { \pmb { x } } } ( \hat { \pmb x } _ { t } * _ { 0 } | \pmb x ) } { \mathbb { E } } [ \mathcal { M } ^ { y } ( \hat { \pmb x } ( 0 ) ) ] \bigg ]
$$

Yet, this expectation is over probabilistic paths whose randomness is due to the noise ϵ of the forward pass and the Brownian motion $d \bar { \mathbf { \boldsymbol { w } } } _ { t }$ at each reverse step t that are independent. Hence, by the law of the unconscious statistician:„

$$
- \nabla _ { \pmb { x } } \bigg [ \underset { p _ { \pmb { \theta } _ { \pmb { x } } } ( \hat { \pmb x } _ { t } * _ { 0 } | \pmb x ) } { \mathbb { E } } \Big [ \underset { \ b { \theta } _ { \pmb { \theta } _ { \pmb { x } } } ( \hat { \pmb x } _ { t } \ast _ { 0 } | \pmb x ) } { \mathbb { E } } \Big ] = - \nabla _ { \pmb { x } } \bigg [ \underset { p _ { r } ( \epsilon , d \bar { \pmb w } _ { t } * , \dots , d \bar { \pmb w } _ { 0 } ) } { \mathbb { E } } \big [ \underset { \pmb { \theta } _ { \pmb { x } } } { \mathbb { M } } ^ { y } ( \hat { \pmb x } ( 0 ) ) \big ] \bigg ]
$$

where $p _ { r } ( \epsilon , d \bar { \pmb { w } } _ { t ^ { \ast } } , . . d \bar { \pmb { w } } _ { 0 } )$ denotes the joint distribution of the noise ϵ and the Brownian motion vectors in the reverse pass. Note that ${ \hat { \pmb x } } ( 0 )$ on the RHS denotes the output obtained by invoking the $D B P$ pipeline with some assignment of these random vectors on the sample x. As earlier, we can interchange the derivation and integration, obtaining:

$$
\underline { { - } } _ { d \bar { w } _ { t } \ast } \mathbb { E } _ { \mathbf { \Phi } } \left[ \nabla _ { \mathbf { x } } [ \mathcal { M } ^ { y } ( \hat { \mathbf { x } } ( 0 ) ) ] \right] .
$$

Let $G ^ { x }$ denote the random variable that is assigned values from $\nabla _ { \pmb { x } } \mathcal { M } ^ { y } ( \hat { \pmb { x } } ( 0 ) )$ , where ${ \hat { \mathbf { x } } } ( 0 )$ is as described above, and denote its covariance matrix by $\Sigma _ { G ^ { \alpha } }$ . Essentially, we are interested in $\mathbb { E } [ G ^ { x } ]$ If we define $\tilde { \nabla } _ { x }$ as:

$$
\widetilde { \nabla } _ { \pmb { x } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \nabla _ { \pmb { x } } [ \mathcal { M } ^ { y } ( \hat { \pmb { x } } ( 0 ) _ { n } ) ]
$$

where each ${ \hat { \pmb x } } ( 0 ) _ { n }$ is the output of DBP invoked with a certain random path $( \epsilon ^ { n } , d \bar { w } _ { t ^ { * } } ^ { n } , . . d \bar { w } _ { 0 } ^ { n } ) \stackrel { i . i . d } { \sim }$ $p ( \epsilon , d \bar { \pmb { w } } _ { t ^ { * } } , . . d \bar { \pmb { w } } _ { 0 } )$ and N is a sufficiently large number of samples. Then due to the central limit theorem, $\begin{array} { r l } { \tilde { \nabla } _ { x } \longrightarrow \mathcal { N } ( \mathbb { E } [ G ^ { x } ] , \frac { \Sigma _ { G ^ { x } } } { N } ) } \end{array}$ . That is, propagating multiple copies through $D B P$ and then averaging the loss gradients computes the required gradient for forcing DBP to alter its output distribution. The larger the number N of samples is, the lower the variance of the error, as can be seen above. Note that the adaptive attack operates exactly in this manner, assuming it uses $\mathcal { M } ^ { y }$ as the loss it minimizes (but the soundness of the approach generalizes intuitively to any other loss with the same objective over the classifier’s logits), proving our claim. □

## D Details on Our DiffGrad Gradient Module

In Appendix D.1, we identify key challenges and flaws in prior work, detailing how DiffGrad systematically resolves each one. Then, in Appendix D.2, we empirically demonstrate the impact of these flaws on $D B P ^ { \circ } { \bf s }$ gradients when left unaddressed, reinforcing their practical significance and validating the need for our reliable module. Appendix D.3 presents the pseudo-code for our memory-efficient checkpointing algorithms, offering a transparent blueprint for reproducible and correct gradient-enabled purification. Finally, in Appendix D.4, we verify the correctness of our implementation, confirming that the gradients computed by DiffGrad match those produced by a true differentiable purification pipeline.

## D.1 The Challenges of Applying Checkpointing to DBP and How DiffGrad Tackles Them

## D.1.1 High-Variance Gradients

The challenge of high-variance gradients due to insufficient EOT samples is introduced in §3.2. We defer its empirical analysis to Appendix D.2, where we quantify the gradient variability under different sample counts. Practical implications for performance and throughput—enabled by DiffGrad’s design—are explored in the ablation study in Appendix I. Accordingly, we focus below on the remaining implementation issues.

## D.1.2 Time Consistency

Rounding issues may emerge when implementing checkpointing over torchsde solvers: Given the starting time $t ^ { * }$ , the solver internally converts it into a PyTorch tensor, iteratively calculating the intermediate outputs by adding negative time increments to this tensor. On the other hand, checkpointing requires re-calculating the intermediate steps’ samples during back-propagation. If this code is oblivious to torchsde’s conversion of the initial time into a PyTorch tensor, it will continue to treat it as a floating point number, updating it with increments of the time step to obtain each intermediate t used to re-compute the dependencies. The PyTorch implementation does not aim for 100% accuracy, leading to minute discrepancies in the current value of t compared to pure floating-point operations. These inaccuracies accumulate over the time horizon, potentially severely affecting the gradients. Instead, we ensure both the solver and checkpointing code use the same objects (either floating points or tensors).

## D.1.3 Reproducing Stochasticity

CuDNN-Induced Nondeterminism. We found that even in deterministic purification pipelines, subtle nondeterminism from PyTorch’s backend can cause discrepancies between forward and backward propagations. Specifically, CuDNN may select different convolution kernels during the recomputation of intermediate states in checkpointing, leading to minute numerical drift.

To eliminate this, DiffBreak explicitly forces CuDNN to behave deterministically, guaranteeing consistent kernel selection and numerical fidelity. While this source of error is subtle, it is measurable:

for $D i f f u r e$ with $t ^ { * } = 0 . 1$ and $d t = 0 . 0 0 1$ , we observed raw and relative gradient discrepancies of $1 e - 4$ and $7 . 2 8 e \mathrm { ~ - ~ } 5$ when all other issues have been addressed. With this fix, the discrepancy dropped to exactly zero. Although less severe than other sources of error (e.g., time inconsistency or missing gradient paths), this level of precision is critical in setups where gradients are small to prevent sign changes that can affect optimization trajectories. Our inspection of prior work’s code reveals this drift was not previously addressed.

Noise Sampler for Guidance Robustness. While the contribution of the stochastic component of dxˆ often cancels analytically in vanilla VP-SDE-style purification (see §2), meaning it does not affect the gradient and therefore is not required to be accurately reproduced during checkpointing, this is not guaranteed in all variants. In particular, many guidance-based schemes—see §3.2—integrate stochastic terms within the guidance function itself. That is, the guidance function may incorporate stochasticity. When this occurs, gradients become sensitive to noise realizations, and failure to preserve them could break correctness. Yet, standard checkpointing only stores the intermediate ${ \hat { \mathbf { x } } } ( t ) { \mathrm { s } }$ . Hence, ${ \hat { \mathbf { x } } } ( t + d t )$ will differ between the propagation phases as dxˆ is computed via different random variables.

We restructure the logic computing dxˆ: We define a function calc_dx that accepts the noise as an external parameter and utilize a Noise Sampler N S, initialized upon every forward propagation. For each t, N S returns all the necessary random noise vectors to pass to calc_dx. Instead of ${ \hat { \mathbf { x } } } ( 0 )$ only, forward propagation also outputs $\dot { N } S$ and a state S containing all ${ \hat { \mathbf { x } } } ( t ) \mathbf { s }$ . These objects are used to restore the path in backpropagation.

The memory cost of our design is $\begin{array} { r } { \mathcal { O } \big ( \frac { t ^ { * } } { \vert d t \vert } \big ) } \end{array}$ , storing only N S, and all ${ \hat { \mathbf { x } } } ( t ) s ,$ , each with a negligible footprint $( \mathcal { O } ( 1 ) )$ compared to graph dependencies [20].

## D.1.4 Guidance Gradients

In schemes that involve guidance, it is typically obtained by applying a function $\scriptstyle { \pmb { g } } _ { f n }$ to ${ \hat { \mathbf { x } } } ( t )$ at each step (note that $\scriptstyle { \pmb { g } } _ { f n }$ may involve stochasticity—see Appendix D.1.3—as is often the case in GDMP [40]). For instance, in Guided-DDPM [40], the original sample x is used to influence the reverse procedure to retain key semantic information, allowing for an increased budget $t ^ { * }$ to better counteract adversarial perturbations. Effectively, it modifies eq. (8) describing the reverse pass of DDPM as:

$$
d \hat { \pmb { x } } = \frac { 1 } { \sqrt { 1 - \beta _ { i } } } ( ( 1 - \sqrt { 1 - \beta _ { i } } ) \hat { \pmb { x } } _ { i } + \beta _ { i } \pmb { s } _ { \theta } ( \hat { \pmb { x } } _ { i } , i ) ) - \boldsymbol { s } \beta _ { i } \nabla _ { \hat { \pmb { x } } _ { i } } G C ( \hat { \pmb { x } } _ { i } , \pmb { x } ) + \sqrt { \beta _ { i } } \boldsymbol { z } _ { i }
$$

where GC is a guidance condition (typically a distance metric), each step minimizes by moving in the opposite direction of its gradient, while the scale s controls the guidance’s influence.

That is, in the specific case of Guided-DDPM, $g _ { f n } ( \hat { \pmb x } ( t ) ) \equiv - \nabla _ { \hat { \pmb x } ( t ) } G C ( \hat { \pmb x } ( t ) , \pmb x )$ , where $_ { G C }$ is a distance metric. Nonetheless, other choices for ${ \_ { g _ { \mathscr { f n } } } }$ may be employed in general. Typically, as the goal of the guidance is to ensure key information from the original sample x is retained, $_ { g \_ f n }$ will also directly involve this x in addition to xˆptq (e.g., GC above). Yet, a naive implementation would back-propagate the gradients to x by only considering the path through ${ \hat { \mathbf { x } } } ( t )$ . Yet, when $_ { g \_ f n }$ relies on a guide constructed from x to influence ${ \hat { \mathbf { x } } } ( t )$ (e.g., guide ” x in Guided-DDPM, it creates additional paths from x to the loss through this guide at each step t. Accordingly, DiffGrad augments the process to include the gradients due to these paths. In the general case, this guide may not be identical to x but can rather be computed based on x or even completely independent (in which case no guidance gradients are collected). DiffGrad captures this nuance through an abstract function $\mathbf { g } _ { - }$ aux that, given x, outputs the guide. Similar to eq. (6), we have that for each t, the gradient of any function $F$ applied to xˆp0q w.r.t. guide due to the path from ${ \hat { \mathbf { x } } } ( t + d t )$ to xˆp0q is given by:

$$
\nabla _ { g u i d e } ^ { t } F = \nabla _ { g u i d e } \langle \hat { \pmb { x } } ( t + d t ) , \nabla _ { \hat { \pmb { x } } ( t + d t ) } F \rangle\tag{9}
$$

as ${ \hat { \mathbf { x } } } ( t + d t )$ is a function of guide. Recall that we are interested in F , which is the loss function over the classifier’s output on ${ \hat { \mathbf { x } } } ( 0 )$ . The gradient of $F$ w.r.t. guide is a superposition of all these paths’ gradients, given as:

$$
\nabla _ { g u i d e } F = \sum _ { t } \nabla _ { g u i d e } ^ { t } F\tag{10}
$$

By the chain rule, $F ^ { \prime } { \boldsymbol { \mathrm { s } } }$ gradient w.r.t x due to the guidance paths, which we denote as $\nabla _ { x } ^ { g } F _ { ; }$ , is:

$$
\nabla _ { \pmb { x } } ^ { g } F = \nabla _ { \pmb { x } } \langle g \pmb { u } i d e , \nabla _ { g u i d e } F \rangle\tag{11}
$$

As a convention, for an unguided process or when guide and x are not related, we define the gradient returned from eq. (11) as $\check { \nabla } _ { x } ^ { g } F \equiv \mathbf { \hat { 0 } }$ , preserving correctness in general. Since x in this guided scenario traverses both the guidance and standard purification paths, the final gradient is the sum of both components.

Finally, automatic differentiation engines, by default, generate gradients without retaining dependencies on the inputs that produced them. Thus, when the guidance itself is in the form of a gradient as in the case of Guided-DDPM or other potential alternatives, its effects will not be back-propagated to x through any of the two paths described above, despite our proposed extensions. DiffGrad alters this behavior, retaining the dependencies of such gradient-based guidance metrics as well.

## D.1.5 Proper Use of The Surrogate Method [24]

As noted in §3.2, DiffHammer [41] adopts the surrogate method (originally proposed by Lee and Kim [24]) for gradient computation, as acknowledged in their Appendix C.1.1. This approach approximates $D B P ^ { \circ } { \bf s }$ gradients by replacing the standard fine-grained reverse process $( \mathbf { e } . \mathbf { g } . , d t =$ 0.001) with a coarser one $( \mathrm { e . g . , } \bar { d t } = 0 . 0 1 )$ , thereby enabling memory-efficient differentiation using standard autodiff. For instance, when using a diffusion time $t ^ { * } = 0 . 1$ for DBP (see §2), the finegrained process would require 100 steps, which is computationally infeasible (memory-exhaustive) for gradient computation without checkpointing. In contrast, the coarse-grained process would only require 10 steps, which is possible.

Specifically, the forward pass is computed using the closed-form solution from eq. (2), while the reverse pass uses dt , enabling efficient gradient computation. Notably, this replacement only occurs for the purpose of updating the attack sample (i.e., computing the gradients). In contrast, when evaluating the generated AE, it is finally purified using the standard fine-grained process that uses dt. Although this yields only approximate gradients for a slightly different process, Lee and Kim [24] found it effective and more accurate than other approximations like the adjoint method [25] (despite remaining slightly suboptimal—see Appendix G).

However, a manual inspection of DiffHammer’s code reveals that their implementation of this surrogate method contains a critical flaw. Rather than recomputing the reverse process with coarse dt as intended, they first run the forward pass with the standard dt, storing intermediate states. Then, during backpropagation, they attempt to backpropagate gradients using checkpoints spaced at dts intervals, but instead reuse the stored dt-based forward states. This leads to a mismatch: gradients intended for steps of size dts are applied to states computed with dt, effectively disconnecting the computation graph and introducing significant gradient error. In practice, this behaves like a hybrid between checkpointing and BPDA [2], where gradients are approximated around fixed anchor points without correctly tracking the reverse dynamics.

As we show in §4.2, this flawed implementation leads to significant attack performance degradation compared to the full gradients. It illustrates the fragility of gradient computation in diffusion defenses and further motivates the need for our reliable DiffGrad module.

## D.2 Demonstrating The Effects of Incorrect Backpropagation

We empirically demonstrate the impact of the identified flaws on DBP’s gradients in Fig. 2 and briefly explain each result below. As numerical results for CuDNN-induced nondeterminism (i.e., unhandled stochasticity) are already presented in Appendix D.1.3, we focus here on the remaining four issues detailed in Appendix D.1:

(a) Effect of insufficient EOT samples. Fig.2a illustrates the impact of using too few EOT samples when computing gradients—see§3.2 and Appendix D.1.1. While prior works [41, 20, 24, 26, 30] typically use 10–20 samples, such low counts, while somewhat effective, may introduce significant variance, resulting in noisy gradients and suboptimal attacks.

To quantify this, we run the following experiment on a CIFAR-10 sample x purified with DiffPure $( t ^ { * } { = } 0 . 1 , \dot { d } t { = } 1 0 ^ { - 3 } )$ . For each EOT count N (shown on the x-axis), we generate N purified copies, compute the average loss, and backpropagate to obtain the EOT gradient. This is repeated 20 times to yield 20 gradients $g _ { 1 } ^ { N } , \ldots , g _ { 2 0 } ^ { N }$ for each N. We then define:

$$
g _ { d } ^ { m e a n } ( N ) = \frac { 1 } { 2 0 } \sum _ { i = 1 } ^ { 2 0 } m a x \parallel g _ { i } ^ { N } - g _ { j } ^ { N } \parallel _ { 2 }
$$

This metric reflects the worst-case deviation between gradients under repeated EOT sampling. By the central limit theorem (see Appendix C), the variance of EOT gradients decays with N, and thus $g _ { d } ^ { \mathrm { m e a n } } ( N )  0$ as $N  \infty$ . The curve in Fig. 2a captures this decay and lets us identify a threshold where variance becomes acceptably low.

The plot confirms that gradient variance decreases sharply with the number of EOT samples. While N “ 10—the typical choice in prior work—is somewhat effective and viable under resource constraints, it remains suboptimal. We observe that variance continues to drop until around $N = 6 4$ after which it plateaus. This suggests that although larger N values like 128 yield marginally better stability, the bulk of the benefit is realized by $N \stackrel { - } { = } 3 2 – 6 4$ . Thus, $N = 1 0$ is a practical baseline, but $N \geqslant$ 64 should be preferred when accuracy is critical and compute allows, coroborating our claims in §3.2.

<!-- image-->

<!-- image-->  
(a) Effect of insufficient EOT samples

(b) Effect of time inconsistency  
<!-- image-->  
(c) Effect of ignoring guidance gradients

<!-- image-->  
(d) Effect of incorrect surrogate gradients  
Figure 2: Effects of the identified flaws in $D B P ^ { \prime } { \bf s }$ backpropagation. Each subfigure visualizes a specific error source.

(b) Effect of time inconsistency. As detailed in §3.2 and Appendix D.1.2, rounding inconsistencies can emerge when using checkpointing with torchsde solvers. These stem from how the solver internally handles time as a $P y$ Torch tensor, which may diverge subtly from floating-point representations if not carefully synchronized. While such discrepancies are negligible for short purification paths (small $t ^ { * } )$ , they accumulate significantly over longer trajectories—particularly at standard DBP settings like $t ^ { * } { = } 0 . 1$ with $d t { = } 1 \bar { 0 } ^ { - 3 } \ ( \mathrm { i . e }$ ., 100 reverse steps), as in $D i f f u r e$ on CIFAR-10.

To quantify this effect, we purify a fixed CIFAR-10 sample x using values of $t ^ { * }$ from 0.005 to 0.1 in steps of 0.005. For each $t ^ { * }$ , we compute gradients using DiffGrad under two conditions: (1) without correcting the inconsistency (yielding $\mathbf { \ } _ { { g } _ { n f } } )$ , and (2) with the issue fixed (yielding $g _ { f } )$ , both using the same stochastic path. We then compute two metrics: (1) $g _ { d } = \| g _ { f } - g _ { n f } \| _ { 2 }$ (absolute error), and $\begin{array} { r } { ( 2 ) g _ { e } = \frac { \| g _ { f } - \pmb { g } n f \| _ { 2 } } { \| \pmb { g } _ { f } \| _ { 2 } } } \end{array}$ (relative error). These quantify the divergence introduced by uncorrected rounding—both in magnitude and in proportion to the true gradient—and illustrate its increasing severity with longer reverse trajectories.

Fig. 2b confirms that even minute rounding discrepancies in time handling, if uncorrected, can completely corrupt gradients over longer diffusion trajectories. Both the absolute error $g _ { d }$ (blue) and the relative error $g _ { e }$ (orange) grow rapidly with $t ^ { * }$ . By $t ^ { * } \geqslant 0 . 0 3 , g _ { e }$ exceeds 90%, and for the typical setting of $t ^ { * } = 0 . 1$ , the relative error reaches $1 0 0 \dot { \% } .$ , indicating that the gradients are nearly orthogonal to the correct direction. This validates our claim that unpatched checkpointing introduces severe errors and demonstrates that our DiffGrad fix is essential for correct gradient computation in practical DBP setups.

(c) Effect of ignored guidance gradients. We evaluate the impact of neglecting guidance gradients, a critical flaw in current attacks against Guided DBP defenses (e.g., GDMP, MimicDiffusion)—see §3.2 and Appendix D.1.4. As explained earlier, guidance introduces additional gradient paths from the input x to ${ \hat { \mathbf { x } } } ( t )$ , typically requiring second-order derivatives to capture. Existing approaches ignore these paths, leading to incomplete gradients and suboptimal attacks. To our knowledge, DiffGrad is the first system to correctly handle them.

To quantify this effect, we purify a fixed CIFAR-10 sample x using GDMP (setup in Appendix F) while varying the purification horizon $t ^ { * } \in \{ 0 . 0 0 6 , 0 . 0 \bar { 1 } 2 , \ldots , 0 . \bar { 0 3 } 6 \}$ . For each $t ^ { * }$ , we compute gradients using DiffGrad with and without accounting for guidance—yielding $g _ { f }$ and $\scriptstyle g _ { n f }$ , respectively—under the same stochastic path. We then compute the absolute and relative gradient error (gd, $g _ { e } )$ as in previous experiments.

The findings in Fig. 2c clearly demonstrate the severe impact of neglecting guidance gradients in Guided $D B P$ setups. Both the raw error $( g _ { d } )$ and the relative error $( g _ { e } )$ increase steeply with $t ^ { * } .$ , confirming that as the purification path grows longer, the omitted gradient paths—stemming from guidance dependencies—dominate the overall gradient signal. Most notably, $g _ { e }$ exceeds $1 0 ^ { 4 }$ at ${ t ^ { * } } = 0 . 0 3 6 { , }$ indicating that the gradient used in prior works is essentially meaningless for optimization, as it no longer aligns with the true direction of steepest descent.

These findings directly explain the poor attack success rates previously observed on guided defenses like GDMP and MimicDiffusion, and reinforce why our attacks, which correctly account for these guidance gradients via DiffGrad, achieve drastically superior performance—see §4.2 and §6. In short, the guidance mechanism, when improperly handled, not only fails to help but actively hinders attack effectiveness by yielding incorrect gradient signals.

(d) Effect of incorrect surrogate gradients. As explained in §3.2 and Appendix D.1.5, prior work [41] uses the surrogate method [24] to enable efficient gradient computation for DBP. However, we identify a critical flaw in DiffHammer’s [41] implementation of this method that substantially compromises attack performance. This flaw not only leads to inflated robustness estimates for $D B P ,$ but also to misleading conclusions about the relative effectiveness of the standard gradientbased attacks versus DiffHammer’s proposed enhancements—see §4.2. Specifically, as detailed in Appendix D.1.5, DiffHammer mismatches the time discretization used in the forward and backward passes: the forward pass computes intermediate denoised states ${ \hat { \mathbf { x } } } ( t )$ using a fine step size $d t ,$ , while the backward pass attempts to propagate gradients at a coarser granularity $\bar { d t } > d t$ using those same (misaligned) forward states. This inconsistency leads to gradients being applied to states generated under a different dynamics, corrupting the computation and severely degrading the attack.

To validate this, we conduct a final experiment: For three different values of $\begin{array} { r l } { d t } & { { } \in } \end{array}$ t0.001, 0.005, 0.01u and fixed $t ^ { * } = 0 . 1$ , we purify a fixed CIFAR-10 sample x. For each $d t ,$ we evaluate four surrogate variants using dt P tdt, 2dt, 5dt, 10dtu. For every configuration, we compute $\scriptstyle g _ { n f }$ using DiffHammer’s flawed surrogate implementation and $g _ { f }$ using the correct implementation. As before, we report both $g _ { d }$ and $g _ { e }$ to quantify the absolute and relative gradient error due to the mismatch between dt and dts . Each experiment is repeated 10 times, and we report the mean values.

The results in Fig. 2d clearly validate our claim: the gradient errors induced by DiffHammer’s flawed surrogate implementation grow substantially with increasing $\bar { d t } / d t$ , saturating quickly even for modest mismatches. Both absolute error $g _ { d }$ and relative error $g _ { e }$ consistently worsen across all dt configurations, indicating that gradients are being applied to the wrong points in the computation graph.



Crucially, DiffHammer uses $d t = 0 . 0 1$ and $d t = 2 d t = 0 . 0 2$ , a setup which already produces substantial degradation $( g _ { d } \approx 3 , g _ { e } \approx 1 )$ , confirming that their reported results rely on gradients that diverge significantly from the correct ones. These errors propagate into the attack, weakening its effectiveness and misleadingly suggesting that first-order attacks are inherently inferior. This experiment conclusively demonstrates that DiffHammer’s surrogate misuse is not a minor detail, but a critical bug that undermines their central claims.

We note that DiffGrad also provides a correct implementation of the surrogate method (despite the full correct gradient being the main method considered in the paper).

## D.3 Pseudo-Code for Our Memory-Efficient Gradient-Enabled Purification with DiffGrad

Forward Propagation. DiffGrad’s forward propagation logic is in Algorithm 1. The code in blue is optional, pertaining to the use of guidance. We highlight in red the portions that differ from standard forward propagation. First, we generate the guidance guide from x (line 1) and disable all graph dependency storage (line 2), enabling our code to run efficiently without attempting to store graphs that will lead to memory failures. Afterward (lines 3-5), we initialize $\pmb { S }$ as an empty list and draw a random seed that is then used to invoke the abstract function init_noise_sampler, which returns a noise sampler that provides a reproducible random path for the backpropagation phase (see §3.2). After the input is diffused (lines 6-8) via eq. (2), lines 9-15 correspond to the reverse pass: At each step t (effectively i), xˆ (that now represents ${ \hat { \mathbf { x } } } ( t ) )$ is first appended to $s ,$ which will eventually contain all such intermediate outputs (line 10). The noise provided by N S for the current step i is then retrieved (line 11) and used to compute dxˆ (line 12). dxˆ is then added to xˆ so that its current value becomes ${ \hat { \mathbf { x } } } ( t + d t )$ . This repeats until ${ \hat { \mathbf { x } } } = { \hat { \mathbf { x } } } ( 0 )$ . Unlike the naive implementation, we only store the intermediate results. For efficiency, we also avoid saving the random noise for each step i but utilize N S to reproduce those variables on demand. Before termination, we re-enable dependency storage (line 16) to ensure our code does not interfere with the execution of any other modules. Finally, xˆp0q is returned together with the state S and the sampler N S, which are stored internally for reproducibility during backpropagation.

Algorithm 1 Differentiable Purification with DiffGrad — Forward Propagation   
Require: Sample x, Score model sθ, Optimal diffusion time $t ^ { * } ,$ , step size dt, Noise scheduler $\beta ,$   
Reverse diffusion function calc_dx, Noise sampler initializer init_noise_sampler, Guidance   
condition ${ \mathrm { g r n } } ,$ Guidance scale s, Auxiliary guidance extractor g_aux   
1: steps $\textstyle \gets { \bigg | } { \frac { t ^ { * } } { d t } } { \bigg | } ,$ , guide Ð g_auxpxq /\* Calc. #steps and init. guide \*/   
2: disable_dependenciespq   
$\cdot$ Dependencies enabled during forward   
propagation. Disable them. \*/   
3: $\mathbf { S } \gets \prod$ /\* Saved state (will eventually hold all   
intermediate reverse steps’ outputs). \*/   
4: seed Ð random_seedpq /\* Seed to initialize noise path \*/   
5: NS Ð init_noise_samplerpseedq /\* Reproducible sampler \*/   
6: α Ð calc_alphapβq /\* Calculate α factors from eq. (2) \*/   
7: Draw $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \pmb { I } _ { d } )$   
8: $\hat { x } \gets \sqrt { \alpha ( t ^ { * } ) } x + \sqrt { 1 - \alpha ( t ^ { * } ) } \epsilon$ /\* Diffuse according to eq. (2) \*/   
9: for i Ð steps, steps ´ 1, ..., 1 do   
10: S.appendpxˆq /\* Set $\cdot$ . \*/   
11: step_noise Ð NS.samplepiq $/ { * }$ Sample the random noise used to   
calculate dxˆ at step i. \*/   
12: dxˆ Ð calc $\begin{array} { r } { \mathbf { d x } ( \hat { { \boldsymbol x } } , s _ { \theta } , i , d t , \beta , } \end{array}$ /\* Calc. dxˆ according to   
13: step_noise, gfn, s, guideq eq. (4) \*/   
14: xˆ Ð xˆ \` dxˆ $/ { } ^ { * }$ Update $\hat { \mathbf { x } } = \hat { \mathbf { x } } ( t + d t )$ . \*/   
15: end for   
16: enable_dependenciespq /\* Re-enable dependencies. \*/   
17: return xˆ, S, NS, guide

DBP Parallelism. Although Algorithm 1 is presented for a single purification path, DiffGrad natively supports parallel purification of N stochastic copies to potentially obtain a significant speedup when computing higher-quality EOT gradients—see §3.2. For simplicity, we abstracted away the batch logic in the pseudo-code. In practice, prior to line 1, the input x is replicated N times (N is an additional argument that can be provided to Algorithm 1). Line 4 returns N random seeds, and the resulting sampler NS manages N reproducible random paths. Line 7 draws N distinct noise samples to generate N diffused versions of x, and all subsequent steps operate copy-wise in parallel across these N instances.

Backpropagation. DiffGrad’s backpropagation logic is in Algorithm 2. Similar to earlier, red text refers to operations that deviate from traditional backpropagation, while blue lines are optional (guidance-related). In addition to the usual gradient grad w.r.t. xˆp0q, the inputs include multiple parameters normally exclusive to forward propagation, as they are required to re-calculate the dependencies. Additionally, the algorithm accepts the saved state ${ \dot { \boldsymbol { S } } } ,$ and the same noise sampler N S to retrieve the stochastic path of the forward propagation. Before providing details, we note that by definition @A, $\pmb { { \cal B } } \in \mathbb { R } ^ { d }$ , it holds that:

$$
\langle A , B \rangle = \sum _ { d } A \odot B
$$

where d denotes the element-wise product. Therefore, in order to calculate the gradients w.r.t. xˆptq and guide as described in eq. (6) and eq. (9), we may define an objective at each step t as:

$$
O b j _ { t } = \sum _ { d } ( \hat { \pmb { x } } ( t + d t ) \odot \nabla _ { \hat { \pmb { x } } ( t + d t ) } F )\tag{12}
$$

and take its gradient w.r.t. the two elements of interest above, which explains the steps in our pseudo-code in Algorithm 2.

Algorithm 2 Differentiable Purification with DiffGrad — Backpropagation   
Require: Loss gradient grad w.r.t xˆ0, Sample x, Score model $\cdot$ , Optimal diffusion time $\cdot$ , step   
size dt, Noise scheduler β, Reverse diffusion function calc_dx, State $\_$   
Noise sampler NS, Auxiliary guidance input guide, Guidance function ˇ ˇ $\mathbf { g _ { f n } }$ , Guidance scale s   
1: $\textstyle s t e p s \gets \left| { \frac { t ^ { * } } { d t } } \right|$   
2: g_grad Ð 0 $/ { } ^ { * }$ Init. gradient w.r.t guidance input \*/   
3: for i Ð 1, 2, ..., steps do   
4: xˆ Ð Sris $J ^ { * } \operatorname { S e t } { \hat { x } } = { \hat { x } } ( t )$ \*/   
5: step_noise Ð NS.samplepiq $/ *$ Retrieve noise for step i \*/   
6: enable_dependenciespq   
7: dxˆ Ð calc_dxpxˆ, sθ, i, dt, β,   
8: step_noise, gfn, s, guide)   
9: $\_$ $\_$ \*/   
10: $O b j _ { t } \gets \sum ( \hat { \pmb { x } } _ { + d t } \odot$ gradq /\* Objective due to eq. (12) \*/   
11: disable_dependenciespq   
12: grad $\gets \nabla _ { \hat { \boldsymbol { x } } } O b j _ { t }$ /\* Update gradient w.r.t xˆptq (eq. (6)) \*/   
13: g_grad Ð g_grad \` $\nabla _ { \mathbf { g u i d e } } O b j _ { t }$ /\* Update guide gradient (eq. (10)) \*/   
14: end for   
15: α Ð calc_alphapβqa   
16: grad Ð grad $* \sqrt { \alpha ( t ^ { * } ) }$ /\* Loss gradient w.r.t x (eq. (2)) \*/   
17: g_grad $\check { \gets } \nabla _ { x } \sum$ pguide d g_gradq /\* Guidance gradient w.r.t x (eq. (11)) \*/   
18: grad Ð grad \` g_grad /\* Merge loss and guidance gradients \*/   
19: return grad

The procedure begins by creating a variable g_grad and setting it to 0 (line 2). This will later be used to store the guidance gradients (see Appendix D.1.4). For each time step t (i.e., step i), starting from $t ^ { \prime } = - d t ( i = 1 )$ , the process (lines 3-14) first retrieves ${ \hat { \mathbf { x } } } ( t )$ from the saved state S (line 4) and the corresponding random noise for that step used during forward propagation (line 5) and computes ${ \hat { \mathbf { x } } } ( t + d t )$ , denoted as $\hat { \pmb { x } } _ { + d t }$ (lines 7-9). Importantly, these computations are performed while storing graph dependencies (enabled on line 6 and re-disabled on line 11 to restore the normal execution state). Specifically, during the first step, we calculate ${ \hat { \mathbf { x } } } ( 0 )$ from ${ \hat { \mathbf { x } } } ( - d t )$ . Afterward, we compute the objective $O b j _ { t }$ (line 10) following eq. (12) that allows us to back-propagate the gradient from xˆp0q to ${ \hat { \mathbf { x } } } ( - d t )$ and guide using the stored dependencies, as per eq. (6) and eq. (9). grad is then updated to hold the gradient of the loss function w.r.t. ${ \hat { \mathbf { x } } } ( - d t )$ as desired (line 12), and the gradient of guide due to this guidance path $( { \mathrm { i . e . } }$ ., from guide to the loss due to guide participating directly in the calculation of ${ \hat { \pmb { x } } } ( t + d t ) -$ see Appendix $ { \mathrm { D } } . 1 . 4 )$ is added to g_grad (line 13). This process repeats until grad finally holds the gradient w.r.t. ${ \hat { \mathbf { x } } } ( t ^ { * } )$ and g_grad holds the sum of gradients due to all guidance paths w.r.t. guide (eq. (10)). Note that after the required gradients w.r.t. ${ \hat { \mathbf { x } } } ( t )$ and guide are obtained at each step, the dependencies are no longer needed and can be discarded. This is where our approach differs from traditional backpropagation algorithms, enabling memory-efficient gradient calculations (at the cost of an additional forward propagation in total). At this point (line 14), we have the gradient $\nabla _ { \hat { \pmb { x } } ( t ^ { * } ) } F$ and all is required is to use it to calculate $\nabla _ { x } F$ , which is trivial due to the chain rule since the closed-form solution for a $\hat { { \pmb x } } ( t ^ { * } ) \equiv { \pmb x } ( t ^ { * } )$ from eq. (2) indicates that this is equivalent to $\nabla _ { \pmb { x } } F = \sqrt { \alpha ( t ^ { * } ) } * ( \nabla _ { \pmb { x } ( t ^ { * } ) } F )$ as we compute on line 16. We then calculate the guidance paths’ gradient w.r.t x following eq. (11) on line 17. Finally, we sum both components, returning the precise full gradient w.r.t. x.

DBP Parallelism. Algorithm 2 is written for a single purified copy but in practice operates over a batch of N stochastic instances like Algorithm 1. As the forward pass stores N trajectories, the backward pass propagates gradients independently for each. The operations in the pseudo-code are thus applied copy-wise in parallel across all N instances. Finally, after line 18, the N gradients are averaged, yielding the EOT gradient.

## D.4 Verifying the Correctness of DiffGrad

To ensure the correctness of DiffGrad, all that is required is to verify that each ${ \hat { \mathbf { x } } } ( t + d t )$ computed during backpropagation (line 9 in Algorithm 2) exactly matches the corresponding forward-computed ${ \hat { \mathbf { x } } } ( t + d t )$ (line 14 in Algorithm 1). This equality guarantees that the reconstructed computation graph faithfully mirrors the one produced by standard autodiff engines during their normal (noncheckpointed) operation, ensuring exact gradient recovery since we use them to perform the necessary backpropagation between each xˆ $( t + d t )$ and xˆptq.

Since our forward pass explicitly stores all intermediate states in $s ,$ we compare each recomputed ${ \hat { \mathbf { x } } } ( t + d t )$ with its forward counterpart during backpropagation. An exact match confirms that the system is computing precise gradients. We manually validated this for all timesteps.

For guidance gradients, correctness follows from the derivations in Appendix D.1.4. Once incorporated, the same matching procedure ensures their correctness as well.

## E Details on Our Low-Frequency (LF) Adversarial Optimization Strategy

## E.1 Understanding Optimizable Filters

In practice, OFs extend an advanced class of filters, namely guided filters, that improve upon the basic filters discussed in §5. Guided filters employ additional per-pixel color kernels that modulate the distortion at critical points: Since filters interpolate each pixel with its neighbors, they are destructive at edges (intersections between different objects in the image), while the values of non-edge pixels are similar to their neighbors, making such operations of little effect on them. Depending on a permissiveness $\sigma ,$ guided filters construct, for each pixel $( i , j )$ a color kernel $c _ { x , \sigma _ { . } } ^ { i , j }$ of the same dimensionality MˆN as K that assigns a multiplier for each of $( i , j ) \mathbf { \bar { s } }$ neighbors, that decays with the neighbor’s difference in value from $( i , j ) \mathrm { \bf { s } }$ . The output at $( i , j )$ involves calculating the effective kernel $\nu _ { { \bf x } , \kappa , \sigma _ { c } } ^ { i , j } = c _ { { \bf x } , \sigma _ { c } } ^ { i , j } \odot \kappa$ (normalized) which then multiplies $( i , j ) \mathrm { \bf s }$ vicinity, taking the sum of this product. Thus, contributions from neighbors whose values differ significantly are diminished, better preserving information.

Guided filters still employ the same K for all pixels, changing only the color kernel that is computed similarly for all pixels. Thus, to incur sufficient changes, they would also require destructive parameters despite them still potentially performing better compared to their pristine counterparts.

Their parameters are also predetermined, making it impossible to optimize them for a specific purpose. The OFs by Kassis and Hengartner [21] build upon guided filters but differ in two ways: First, instead of using the same K, they allow each pixel to have its own kernel $\kappa ^ { i , j }$ to better control the filtering effects at each point, ensuring visual constraints are enforced based on each pixel’s visual importance. In this setting, ${ \boldsymbol { \kappa } } ^ { * }$ denotes the set including all the per-pixel kernels $\kappa ^ { i , j }$ . Second, the parameters $\theta _ { \kappa * }$ of each filter are learnable using feedback from a perceptual metric (lpips) [56] that models the human vision, leading to an optimal assignment that ensures similarity while maximizing the destruction at visually non-critical regions. To further guarantee visual similarity, they also include color kernels similar to guided filters (see original paper for details [21]).

## E.2 Attack Hyperparameters

Through experimentation, we found that the loss balancing constants $c = 1 0 ^ { 8 }$ for CIFAR-10 and $c = 1 0 ^ { 4 }$ for ImageNet lead to the fastest convergence rates and selected these values accordingly (although other choices are also possible). UnMarker’s filter network architecture for ImageNet is identical to that from the original paper [21]. For CIFAR-10, since the images are much smaller, we found the original architecture unnecessarily costly and often prevents convergence since larger filters group pixels from distant regions together in this case, easily violating visual constraints upon each update, resulting in the lpips condition being violated. Thus, we opt for a more compact network that includes filters with smaller dimensions, which was chosen based on similar considerations to [21], allowing us to explore several interpolation options. The chosen architecture for CIFAR-10 includes 4 filters, whose dimensions are: p5, 5q, p7, 7q, p5, 5q, p3, 3q. We use fixed learning rates of 0.008 for the direct modifier δ and 0.05 for the filters’ weights, optimized using Adam. The remaining hyperparameters were left unchanged compared to [21].

## E.3 Pseudo-Code

The pseudo-code for our low-frequency (LF) strategy (see §5) is in Algorithm 3. Importantly, as each $\kappa _ { b } ^ { i , j } :$ s values should be non-negative and sum to 1, the values for each such per-pixel kernel are effectively obtained by softmaxing the learned weights. Initially, the modifier $\hat { \delta }$ is initialized to 0 and the weights $\{ \widehat \theta _ { \kappa _ { h } ^ { * } } \}$ are selected s.t. the filters perform the identity function (line 1). As a result, the attack starts with $\pmb { x } _ { a d v } = \pmb { x }$ that is iteratively optimized. Similar to C&W [4], we directly optimize $\mathbf { { x } } _ { a d v }$ (i.e., using the modifier δˆ) in the arctanh space, meaning we transform the sample first to this space by applying arctanh (after scaling $\mathbf { { x } } _ { a d v }$ to arctanh’s valid range r´1, 1s) where $\hat { \delta }$ is added and then restore the outcome to the original problem space (i.e., rmin_val, max_vals, which is typically r0, 1s) via the tanh operation. Further details on this method and its benefits can be found in [4]. All other steps correspond to the description brought in §5. Unlike §3.1, we assume the classifier M outputs the logit vector rather than the probabilities (i.e., we omit the sof tmax layer over its output, which me or may not be re-introduced by the loss $\ell ) ,$ as is traditionally done for a variety of adversarial optimization strategies (e.g., C&W [4]) to avoid gradient vanishing. We use the known max-margin loss $[ 4 ] \longrightarrow \ell ( l o g i t s , y ) = l o g i t s [ , : y ] - \mathop { m a x } _ { j \neq y } \left\{ l o g i t s [ , : j ] \right\}$

To average the gradients over multiple (N ) paths as per the adaptive attack’s requirements from §3.1, we generate several purified copies by repeating the sample $\mathbf { \Delta } \mathbf { x } _ { a d v }$ under optimization n times before feeding it into the DBP pipeline (line 9). Here, n corresponds to the maximum number of copies we can fit into the GPU’s memory during a single run. However, as this n may be smaller than the desired N from §3.1 (i.e., number of EoT samples) that allows us to sufficiently eliminate the error in the computed gradient, we use gradient accumulation by only making updates to the optimizable parameters (and then resetting their gradients) every eot_steps iterations (lines 16-19). By doing so, the effective number of used copies becomes n ˚ eot_steps, which can represent any N of choice that is divisible by n. Note that if n is not a divisor of N , we can always increase N until this condition is met, as a larger N can only enhance the accuracy). This also explains why the algorithm runs for max_iters ˚ eot_steps (line 5).

Finally, the condition Cond captures the threat model (either $S P$ or MV— see §2): When the logits for the batch of n copies are available together with the target label $y ,$ Cond outputs a success decision based on whether we seek misclassification for the majority of these purified copies or a single copy only. Note that, as explained in §4.2, we take the majority vote over the maximum number of copies we can fit into the $G P U \left( { \mathrm { i } } . { \mathrm { e } } . , n \right) $ for MV. As this choice was only made for practical considerations, one may desire to experiment with different configurations wherein another number of copies is used. Yet, this is easily achievable by simply modifying Cond accordingly: For instance, we may augment it with a history that saves the output logits over all eot_steps (during which $\mathbf { { x } } _ { a d v }$ is not updated). Then, the majority vote can be taken over all copies in this window. Note that by increasing the number of eot_steps, we can use as many copies for the majority vote decision as desired in this case. That said, the attack will become significantly slower.

```latex
Algorithm 3 Low-Frequency (LF) Adversarial Optimization
Require: Sample x, Model (classifier) M, DBP pipeline D, Loss function $\ell ,$ True label y of
x, Perceptual loss lpips, lpips threshold $\tau _ { p } ,$ Filter architecture $\prod _ { O F } ^ { B }$ , Balancing constant c,
Iterations max_iters, Success condition Cond, Filter weights learning rate $l r _ { o F }$ , Modifier
learning rate $l r _ { \delta }$ , Number of purified copies n, Number of EoT eot_steps, Input range limits
(min_val, max_val)
1: $\{ \widehat \theta _ { \kappa _ { b } ^ { * } } \} \gets$ identity_weightsp śB q, δˆ Ð 0 /* Initialize attack parameters. */
2: $O p t i m  A d a m ( [ \{ \hat { \theta } _ { \mathcal { K } _ { h } ^ { * } } \} , \hat { \delta } ] , [ l r _ { o r } , l r _ { \delta } ] )$
3: $\mathbf { \Delta } x _ { i n v } \gets$ inv_scale_and_arctanhpx, min_val, max_valq /* Scale x to r´1, 1s and take
arctanh */
4: for i Ð 1 to max_iters ¨ eot_steps do
5: ${ \pmb x } _ { a d v } \gets \hat { \textmd { \textmu } } _ { } ^ { B }$ ptanh_and_scale $\mathbf { \nabla } _ { \cdot } ( \mathbf { x } _ { i n v } + \hat { \delta } ,$ min_val, max_valqq
/*
Generate adversarial input using
new $\{ \widehat \theta _ { \kappa _ { h } ^ { * } } \}$ and $\hat { \delta }$ via (eq. (7))
scaled to rmin_val, max_vals. */
6: $\mathop { d i s t } \gets l p i p s ( { \pmb x } , { \pmb x } _ { a d v } )$ /* Calculate perceptual distance. */
7: $\hat { \pmb { x } } _ { a d v } ^ { 0 }  \bar { \pmb { D } } ( \pmb { \imath }$ repeat $( \pmb { x } _ { a d v } , n ) )$ /* Get purified outputs. */
8: logits $\gets \mathcal { M } ( \hat { \pmb x } _ { a d v } ^ { 0 } )$ /* Compute model output. */
9: if Condplogits, yq and $d i s t \leqslant \tau _ { p }$ then
10: return xadv $/ *$ Success. Return $\mathbf { \Delta } \mathbf { x } _ { d v } .$ . */
11: end if
12: $O b j e c t i v e \gets \ell ( l o g i t s , y ) + c \cdot m a x ( d i s t - \tau _ { p } , 0 )$ /* Compute loss (eq. (7)). */
13: Objective.backwardpq /* Get gradients for parameters. */
14: if i mod eot_steps “ 0 then
15: Optim.steppq /* Update parameters. */
16: Optim.zero_gradpq /* Reset gradients. */
17: end if
18: end for
19: return x /* Failure. Return original x. */
```

In addition to the precise gradient module DiffGrad, our DiffBreak toolkit provides the implementation of our LF strategy as well as various other common methods (e.g., AA and StAdv), to enable robust and reliable evaluations of DBP. All strategies are optimized for performance to speed up computations via various techniques such as just-in-time (JIT) compilation. Our code is available at https://github.com/andrekassis/DiffBreak.

## F Additional Details on Systems & Models

WideResNet-28-10 and WideResNet-70-16 [52] are used for CIFAR-10, and ResNet-50 [18], WideResNet-50-2, and DeiT-S [14] for ImageNet, similar to [30, 20]. For VP-SDE DBP (Diff-Pure) [30], the DMs [12, 36] are those from the original work. We also experiment with the Guided-DDPM (see §3.2), GDMP [40], due to its SOTA robustness, using the author-evaluated DMs [12, 19]. The settings match the original optimal setup [40, 30]: For Diffpure, t˚ “0.1 for CIFAR-10 and t˚“0.15 for ImageNet. For GDMP, a CIFAR-10 sample is purified m“4 times, with each iteration running for 36 steps (t˚“0.036), using MSE guidance [40]. ImageNet uses 45 steps (m“1) under DDPM-acceleration [12] with SSIM guidance [43].

## G One-Shot DBP Baseline Comparisons Against DiffGrad’s Accurate Gradients

As noted in §3, $D B P ^ { \prime } { \bf s }$ robustness stems from two flaws: inaccurate gradients and flawed evaluation. As our work offers enhancements on both fronts, we evaluate each factor separately. Here, we isolate the gradient issue by re-running prior experiments under the same (flawed) one-shot evaluation protocol, but with accurate gradients via our DiffGrad module and compare the results to those from the literature. We restrict our attacks to 10 optimization steps (with $A A \mathrm { - } \ell _ { \infty } )$ , while prior works often use up to 100, giving them a clear advantage. We evaluate on CIFAR-10 with WideResNet-28-10, using N “ 128 EOT samples as justified in §3.2. For papers reporting several results, we chose their best.

Baselines. To demonstrate the efficacy of exact full gradients, we compare our DiffGrad module against prior gradient approaches. For DiffPure, the adjoint method was originally used [30], while GDMP was initially evaluated using BPDA [2] and a blind variant (ignoring the defense entirely) in Wang et al. [40]. These approximations were later criticized for poor attack performance [41, 26, 20, 24]. More recently, Lee and Kim [24] proposed the surrogate process to approximate the gradients, performing the reverse pass with fewer steps during backpropagation to reduce memory usage, enabling approximate gradients via standard autodiff tools. DiffAttack [20], DiffHammer [41], and Liu et al. [26] proposed checkpointing for memory-efficient full-gradient backpropagation. However, DiffHammer avoids the standard one-shot evaluation (1-evaluation) protocol and reports no results under it, while DiffAttack does not evaluate GDMP and continues to use the adjoint method for DiffPure. We focus here on existing results under the 1-evaluation protocol and defer the discussion of these works’ conceptual flaws (see §1) to §4.2.

Metric. Following prior works, we report robust accuracy (Rob-Acc): the fraction of samples correctly classified after the attack completes and the final adversarial example is purified and evaluated (once).

Results. Table 4 shows our comparison. All methods achieve similar clean accuracy (Cl-Acc) without attacks, with minor variation due to sample selection. Thus, robust accuracy (Rob-Acc) differences reflect the effect of gradient methods.

Our approach significantly outperforms gradient approximations such as Adjoint, Blind, and BPDA, reaffirming their known weaknesses. More notably, despite using only 10 optimization steps, our method reduces Diff-Pure’s Rob-Acc by 14.06% compared to Liu et al. [26], who also use exact gradients, confirming their backpropagation flaws (see §3.2).

Table 4: One-shot $A A \mathrm { - } \ell _ { \infty }$ comparison on CIFAR-10 $( \epsilon _ { \infty } { = } 8 / 2 5 5 )$ : indicates strategy is PGD.
<table><tr><td>Models</td><td>Pur.</td><td>Gradient Method</td><td>Cl-Acc %</td><td>Rob-Acc %</td></tr><tr><td rowspan="5">WideResNet-28-10</td><td rowspan="5">DiffPure [30]</td><td>Adjoint (Nie et al. [30])</td><td>89.02 89.02</td><td>70.64</td></tr><tr><td>DiffAttack (Kang et al. [20])</td><td></td><td>46.88</td></tr><tr><td>Surrogate (Lee and Kim [24])†</td><td>90.07</td><td>48.28</td></tr><tr><td>Full (Liu et al. [26])</td><td>89.26</td><td>62.11</td></tr><tr><td>Full-DiffGrad (Ours)</td><td>89.46</td><td>48.05</td></tr><tr><td rowspan="5"></td><td>Blind (Wang et al. [40])</td><td></td><td>93.50</td><td>90.06</td></tr><tr><td>GDMP [40]</td><td>BPDA (Lee and Kim [24])</td><td>89.96</td><td>75.59</td></tr><tr><td></td><td>Surrogate (Lee and Kim [24])†</td><td>89.96</td><td>24.53</td></tr><tr><td></td><td>Full-DiffGrad (Ours)</td><td>93.36</td><td>19.53</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

## While DiffAttack remains

slightly stronger, the difference is extremely small (1.4%) and can be safely attributed to differing evaluation samples. Importantly, the original gap reported in Kang et al. [20] between DiffAttack and the standard $A A \mathrm { - } \ell _ { \infty }$ dropped by 23.76%, undermining its claimed advantage due to the per-step deviated reconstruction losses and aligning with our theoretical findings in §3.1. As DiffAttack involves broader architectural changes beyond gradient logic, we provide detailed comparisons in §4.2, where we clearly showcase its inferiority.

Our method also slightly outperforms Surrogate for DiffPure (0.23%), but we caution against overinterpreting this small gap: under the flawed one-shot evaluation, several purification paths can still cause misclassification even if the majority yield correct labels. As such, good approximation methods like Surrogate may appear closer in performance than they truly are. To verify the superiority of our DiffGrad’s full gradients over the surrogate approximation against $D i f f P u r e .$ , we thus further compare the two under the realistic MV protocol studied in §4.2, executing $A A \mathrm { - } \ell _ { \infty }$ with the surrogate process to obtain the gradients when attacking the same CIFAR-10 classifier considered in §4.2 (i.e., WideResNet-70-16) and using the same number of samples (N “ 10) over which the majority vote is taken. We find that the surrogate process brings MV.Rob to 43.75% only, whereas our Full-DiffGrad lowers this number to 39.45%, achieving a considerable improvement of 4.3% and unequivocally proving the advantage of exact gradient computations. Finally, for GDMP, our method outperforms Surrogate by 5% even in the one-shot evaluation protocol (see Table 4), largely due to our incorporation of guidance gradients—entirely absent in all prior approaches—highlighting the unique strength of DiffGrad.

All in all, the results demonstrate the fragility of DBP in the face of accurate gradients and highlight the glaring flaws in previous works’ backpropagation. Nonetheless, this one-shot evaluation protocol remains flawed, as previously stated, leading to an inflated robustness estimate. In fact, under more realistic settings (see §4.2), we demonstrate that this gradient-based attack almost entirely defeats DBP when only one sample is used to predict the label (i.e., Wor.Rob).

## H Evaluations with Liu et al. [26]’s Fixed AutoAttack

Liu et al. [26], like DiffHammer [41] and our own analysis, note that evaluating a single purification at attack termination inflates robustness scores. Liu et al. [26] address this by evaluating 20 replicas of the final AE, declaring success if any is misclassified. While similar in spirit to the Wor.Rob metric we consider (see §3.3), this protocol (see §3.3), this protocol Table 5: is more limited: it evalu- is more limited: it evalu- $A A \mathrm { - } \ell _ { \infty }$ comparison on CIFAR-10 under Liu et al. [26]’s ates only at the final step, ates only at the final step, protocol $( \mathbf { i . e . } ,$ Fixed AutoAttack) $( \epsilon _ { \infty } { = } 8 / 2 5 5 )$ .

whereas Wor.Rob evaluates N copies at each attack iteration. Accordingly, Liu et al. [26] group their method with one-shot evaluations, providing a slightly more realistic assessment that leads to results similar

<table><tr><td>Models</td><td>Pur.</td><td>Gradient Method</td><td>Cl-Acc %</td><td>Rob-Acc %</td></tr><tr><td rowspan="3">WideResNet-28-10</td><td>DiffPure [30]</td><td>Full (Liu et al. [26])</td><td>89.26</td><td>56.25</td></tr><tr><td></td><td>Full-DiffGrad Full (Liu et al. [26])</td><td>89.46 91.80</td><td>30.86 40.97</td></tr><tr><td>GDMP [40]</td><td>Full-DiffGrad</td><td>93.36</td><td>10.55</td></tr><tr><td rowspan="2">WideResNet-70-16</td><td>Diff Pure [30]</td><td>Full-DiffGrad</td><td>89.06</td><td>35.16</td></tr><tr><td>GDMP [40]</td><td>Full-DiffGrad</td><td>91.8</td><td>8.59</td></tr></table>

to those attained via PGD [23].

We replicate their setup using DiffGrad (Table 5), running 20 iterations with N “ 10 EOT samples per step and evaluating 20 final replicas. Full denotes the standard AA attack that uses the full exact gradients (i.e., via checkpointing). Under this protocol, our improvements are stark. On WideResNet-28-10, we reduce Rob-Acc by 25.39% and 30.42% for DiffPure and GDMP, bringing final accuracy to 30.86% and 10.55%, respectively. These results confirm the superiority of DiffGrad’s gradients and expose $D B P ^ { \circ } { \bf s }$ realistic vulnerability. We observe similar results on WideResNet-70-16 (not evaluated in [26]).

## I Ablation Study with Different Numbers of Samples for Label Prediction

To assess the impact of the number of evaluation samples $N ,$ we test both single-purification and majority-vote (MV) settings with $N \in \{ 1 , 1 0 , 1 2 8 \}$ . Our goal is to highlight the brittleness of $D B P ^ { \bullet } \mathbf { s }$ standard deployment, which classifies based on a single purified copy. As explained in §3.3, in stateless setups (e.g., phishing, spam, CSAM), adversaries can resubmit identical queries indefinitely. Since randomized defenses like DBP can fail along certain stochastic paths, any non-negligible misclassification probability compounds with repeated attempts—see §3.3. Although $D B P$ assumes this probability is negligible, our proof in §3.1, results in $\ S 4 ,$ and the ablation experiments here contradict this. Furthermore, we wish to showcase the advantages of our proposed majority-vote (MV) setting that strives to mitigate this vulnerability by, instead, classifying based on expectation of the randomized defense.

We evaluate DiffPure and GDMP on CIFAR-10 using WideResNet-70-16 and (our) $A A \mathrm { - } \ell _ { \infty }$ (with $\epsilon = 8 / 2 5 5$ and 100 iterations). For $N = 1$ and $N = 1 0$ , we use 10 EOT samples; for $N = 1 2 8$ we reuse the same 128 samples for both EOT and evaluation. This improves gradient quality, thus strengthening attacks, yet still leads to higher MV robustness, proving the superiority of our proposed method despite the use of more accurate gradients.

For Wor.Rob, clean accuracy (Cl-Acc) remains constant across all N values since we always compute it using a single purified copy $( N ~ = ~ 1 )$ independent of the number N of samples used in the attack. This reflects the actual standard single-purification deployment of DBP, where the defender classifies a single output, while the attacker may retry multiple times. Hence, for Rob-Acc we consider a batch of multiple samples (the corre-

Table 6: $A A \mathrm { - } \ell _ { \infty }$ Performance under various evaluation sample counts
<table><tr><td rowspan="2">Pur.</td><td rowspan="2">#Samples</td><td colspan="2">Wor.Rob %</td><td colspan="2">MV.Rob %</td></tr><tr><td>Cl-Acc</td><td>Rob-Acc</td><td>Cl-Acc</td><td>Rob-Acc</td></tr><tr><td rowspan="4">Diff Pure [30]</td><td>N=1</td><td rowspan="4">89.06</td><td>35.16</td><td>89.06</td><td>35.16</td></tr><tr><td>N=10</td><td>17.19</td><td>91.02</td><td>39.45</td></tr><tr><td>N=128</td><td>17.58</td><td>92.19</td><td>47.72</td></tr><tr><td>N=1</td><td>8.59</td><td>91.8</td><td>8.59</td></tr><tr><td rowspan="3">GDMP [40]</td><td>N=10</td><td rowspan="3">91.8</td><td>7.03</td><td>92.19</td><td>16.8</td></tr><tr><td></td><td></td><td>92.19</td><td></td></tr><tr><td>N=128</td><td>5.47</td><td></td><td>32.81</td></tr></table>

sponding N in that row) and then declare attack success if a single misclassification occurs, revealing the gap between measured and effective robustness. In contrast, MV.Rob clean accuracy varies with N, as the prediction always aggregates over N purified samples, consistent with our proposed deployment.

Table 6 confirms that Wor.Rob consistently declines with N , revealing the illusion of robustness under single evaluation. For instance, DiffPure shows a drop from 35.16% (at N “ 1) to 17.58% at $N = 1 2 8$ (far lower than the inflated numbers reported in previous studies—see Appendix G), confirming that many stochastic paths yield incorrect predictions, as our properly implemented gradient-based attack (see §3.2) lowers the expected classification confidence, making such failure modes more likely. For GDMP, we observe a similar trend.

In contrast, MV.Rob improves with N , rising from 35.16% to 47.72% for DiffPure, and from 8.59% to 32.81% for GDMP. This affirms MV as a more stable and accurate evaluation method that must be adopted as the de facto standard for DBP evaluations in the future.

Yet, despite MV’s benefits, our gradient-based $A A \mathrm { - } \ell _ { \infty }$ attack still degrades robustness, which never exceeds 50%, validating our theoretical finding from §3.1 that such attacks repurpose DBP into an adversarial distribution generator.

Computational Cost. While larger N increases computational overhead, we identify N “ 128 as the max batch size fitting in a single A100 GPU run requiring „ 26.53s for inference, with $N = 1 0$ offering a practical middle ground (6.54s vs. 5.29s for $N = 1 )$ . Hence, for practicality and if throughput is critical, one should opt for $N = 1 0$ . However, in security-critical systems where latency is not crucial, a larger N is favorable. Yet, additional tests (not shown) suggest MV.Rob plateaus near $N = 1 2 8$ . Note that these latencies refer to purification inference alone (classification excluded). During attacks (not standard deployment), there is also the cost of backpropagation. When accounting for the cost of classification and backpropagation, the latency becomes „ 17s for N “ 1 and $N = 1 0$ compared to „ 53s for $N = 1 2 { \bar { 8 } }$ . As explained in §3.2, our design allows purifying multiple stochastic copies of the same sample in one step, in contrast to previous work. This reduces the runtime per EOT gradient from up to N ˆ 17s—where N stochastic purifications are run serially—to just $T _ { N } \mathbf { s } ,$ where $T _ { N }$ is the latency incurred by purifying N samples in a batch, yielding a speedup of up to $N \times 1 7 / T _ { N }$ for single-sample attacks, which amounts to 41.06ˆ for $N = 1 2 8$ allowing for a considerable speedup when the objective is to utilize many EOT samples to obtain accurate gradient estimates. For batch attacks, prior methods purify one stochastic copy per sample per EOT step and require all samples to converge before proceeding to the next batch of inputs, blocking early termination. In contrast, DiffGrad purifies N copies of a single sample in parallel, allowing samples to terminate independently. This greatly improves overall throughput by freeing compute sooner, especially when convergence varies across inputs.

## J Example Attack Images

In Appendix J.1-J.10, we provide a variety of successful attack images that cause misclassification in the rigorous MV setting, generated using our low-frequency (LF) strategy against all systems considered in §5.1. For configurations that were also evaluated against $A A \mathrm { - } \ell _ { \infty }$ under the same MV setting (i.e., using the same sample counts N as used for LF in §5.1), we include successful AEs generated with this method for direct comparison. Specifically, for ImageNet, both the LF attack in §5.1 and the AA attack in §4.2 were evaluated under MV with $N = 8 ;$ hence, we include AA samples for ImageNet directly from §4.2. For CIFAR-10, the AA experiments in §4.2 use the more permissive N “ 10 setting. Therefore, we instead draw samples from the corresponding N “ 128 AA experiments reported in Appendix I to ensure a fair comparison under equal majority-vote conditions. Note that all samples are crafted using the parameters listed in §4.2 and §5.1. That is, $\tau _ { p } = 0 . 0 5$ for LF and $\epsilon _ { \infty } = 8 / 2 5 5$ for AA against CIFAR-10 and $\epsilon _ { \infty } = 4 / 2 5 5$ against ImageNet.

For the configurations that were evaluated against both strategies, we provide two sets of samples: 1) Three triplets containing the original image, the AE generated using AA, and the AE crafted using LF. Importantly, all original samples in this set are inputs for which both methods can generate successful AEs, and we provide these to allow for a direct comparison between the two strategies’ output quality on a sample-by-sample basis. Yet, as AA is inferior to our approach (LF), resulting in the systems retaining robustness on many inputs for which it fails to generate successful AEs under MV, it is essential to inspect $L F { ^ { \circ } } { _ { \mathbf { S } } }$ outputs on such more challenging samples to demonstrate that it still preserves quality despite its ability to fool the target classifiers. Thus, we include a second set of 2) Three successful AEs generated with LF from inputs on which AA fails under MV. For the remaining configurations that were not evaluated against AA under the same MV sample counts from §5.1, we provide six successful AEs generated with LF.

In Appendix J.11, we present attack images generated by the non-norm-bounded StAdv [47] strategy under MV (with the above sample counts N ). This method has demonstrated superior performance to norm-based techniques in the past against DBP [30] even in the absence of the correct exact gradients, indicating it could be a viable attack strategy with our gradient computation fixes, thereby making our LF approach unnecessary. Yet, previous evaluations only considered StAdv against DBP for CIFAR-10 [30]. While we find StAdv capable of defeating all systems (for both CIFAR-10 and ImageNet), it leads to severe quality degradation when used to attack DBP-protected classifiers for high-resolution inputs (i.e., ImageNet), leaving the AEs of no utility. Thus, we deem it unsuitable, excluding it from the main body of the paper accordingly. Further details are in Appendix J.11.

All samples below are originally (without adversarial perturbations) correctly classified.

## J.1 Attack Samples Generated Against CIFAR-10’s WideResNet-70-16 with GDMP Purification

<!-- image-->

<!-- image-->

<!-- image-->  
Figure 3: Successful attacks generated by LF and $A A { - } \ell _ { \infty }$ . Left -original image. Middle - AA. Right - LF.

<!-- image-->

<!-- image-->  
Figure 4: Successful LF attacks on inputs for which $A A \mathrm { - } \ell _ { \infty }$ fails. Left - original image. Right - LF.

## J.2 Attack Samples Generated Against CIFAR-10’s WideResNet-28-10 with GDMP Purification

Note: While WideResNet-28-10 was not evaluated against AA under MV with N “ 128 in Appendix I, we include AA samples here to complement our CIFAR-10 analyses—our primary benchmark for AA-based evaluation. Among the two main purification paradigms considered (GDMP and DiffPure), we randomly selected GDMP for this illustrative example.

<!-- image-->

<!-- image-->

<!-- image-->  
Figure 5: Successful attacks generated by LF and $A A { - } \ell _ { \infty }$ . Left -original image. Middle - AA. Right - LF.

<!-- image-->

<!-- image-->  
Figure 6: Successful LF attacks on inputs for which $A A \mathrm { - } \ell _ { \infty }$ fails. Left - original image. Right - LF.

<!-- image-->

## J.3 Attack Samples Generated Against CIFAR-10’s WideResNet-70-16 with DiffPure Purification

<!-- image-->

<!-- image-->

<!-- image-->  
Figure 7: Successful attacks generated by LF and $A A { - } \ell _ { \infty }$ . Left -original image. Middle - AA. Right - LF.

<!-- image-->

<!-- image-->

<!-- image-->  
Figure 8: Successful LF attacks on inputs for which $A A \mathrm { - } \ell _ { \infty }$ fails. Left - original image. Right - LF.

## J.4 Attack Samples Generated Against CIFAR-10’s WideResNet-28-10 with DiffPure Purification

<!-- image-->

<!-- image-->

<!-- image-->  
Figure 9: Successful attacks generated with LF. Left -original image. Right - LF.

<!-- image-->  
Figure 10: Successful attacks generated by LF and $A A \mathrm { - } \ell _ { \infty }$ . Left -original image. Middle - AA. Right - LF.

<!-- image-->  
Figure 11: Successful LF attacks on inputs for which $A A { - } \ell _ { \infty }$ fails. Left - original image. Right - LF.

<!-- image-->

<!-- image-->  
Figure 12: Successful attacks generated with LF. Left -original image. Right - LF.

## J.7 Attack Samples Generated Against ImageNet’s ResNet-50 with GDMP Purification

<!-- image-->

<!-- image-->  
Figure 13: Successful attacks generated with LF. Left -original image. Right - LF.

<!-- image-->  
Figure 14: Successful attacks generated by LF and $A A \mathrm { - } \ell _ { \infty }$ . Left -original image. Middle - AA. Right - LF.

<!-- image-->  
Figure 15: Successful LF attacks on inputs for which $A A { - } \ell _ { \infty }$ fails. Left - original image. Right - LF.

<!-- image-->  
Figure 16: Successful attacks generated by LF and $A A \mathrm { - } \ell _ { \infty }$ . Left -original image. Middle - AA. Right - LF.

<!-- image-->  
Figure 17: Successful LF attacks on inputs for which $A A { - } \ell _ { \infty }$ fails. Left - original image. Right - LF.

<!-- image-->

<!-- image-->  
Figure 18: Successful attacks generated with LF. Left -original image. Right - LF.

## J.11 Quality Comparison with StAdv

We found StAdv capable of generating outputs that defeat DBP even under MV. However, it is not suitable for targeting DBP-defended classifiers that operate on high-resolution images. The reason is that StAdv performs spatial transformations that relocate the different pixels. Thus, its changes quickly become visible when applied excessively. Due to the considerable stochasticity of DBP (see §4.2), the required displacements (especially in the MV setting) are significant, which in turn can severely impact the quality. For low-resolution inputs (e.g., CIFAR-10), StAdv can still be effective, with the quality degradation remaining unnoticeable due to the size of the images that renders them blurry by default, masking StAdv’s effects. For high-resolution inputs, the degradation is substantial, leaving the outputs useless as stealthiness is a key requirement from practical AEs [21]. StAdv’s successfully misclassified samples (under MV) below prove these claims. We use Full-DiffGrad for backpropagation and run StAdv with its default parameters [47]. When the parameters are changed to better retain quality, StAdv ceases to converge for ImageNet, making it of no use. All provided samples are originally correctly classified.

<!-- image-->

<!-- image-->

<!-- image-->  
Figure 19: StAdv attacks against CIFAR-10’s WideResNet-70-16 with GDMP purification. Left - original image. Right - StAdv.

<!-- image-->

<!-- image-->  
Figure 20: StAdv attacks against ImageNet’s DeiT-S with DiffPure purification. Left - original image. Right - StAdv.



