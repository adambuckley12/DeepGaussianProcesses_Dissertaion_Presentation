# Presentation Script: Deep Gaussian Processes
## 10-Minute Presentation Script

---

## [SLIDE 1 - Title Slide]
**[0:00 - 0:20]**

So my plan is to explain to you what deep gausian processes are. Firstly i'll introduce Gaussian Processes (GPs), how we derive them and then we will see how these limitations in these GPs motivate extending Deep GPs. Ill explain the maths behind deep GPs and then move on tohow we train these models. Lastly, I'll present early results from a credit risk modelling case study.

---

## [Towards Gaussian Processes]
**[0:40 - 1:10]**

So where to start?

Lets suppose we have some dataset D and we want to find a function f that maps our input data to outputs. Supose we start at the most basic approach with linear regression. This works for linear data, but the linearity is too restrictive for many real-world problems. We can extend this using basis functions, where we transform our inputs through fixed functions phi. This gives us more flexibility and allows non-linearity, but we still need to choose how many basis functions to use and what form they should take. 

What if we dont know what basis functions to pick? Is there a way to avoid the need to chose parameters?

---

## [SLIDE 4 - From Basis Functions to GPs]
**[1:10 - 1:40]**

Suppose instead of treating our weights as point estimates, we place a Gaussian prior over them. 

This means that before we see any data, we have a whole range of possible functions that could explain our data, each with a certain probability. The shape of this function distribution is determined by the choice of basis functions and the weight prior.

Notice that this function distribution depends ONLY on the combined effect of the inner product of the basis functions and weight prior, not on either individually. This combined effect is the kernel function k, which captures how similar two function values should be based on their inputs. We then have a kernel matrix K that captures the covariance between function values at all pairs of inputs.

---

## [SLIDE 6 - GP Definition]
**[2:00 - 2:25]**

This brings us to Gaussian Processes. Instead of having a distribution over parameters like in traditional models, we now have a distribution over entire functions.

Formally a GP is defined as a collection of random variables, any finite number of which have a joint Gaussian distribution. We can specify a GP by its mean function m(x) and covariance function k(x, x'). The mean function gives us the expected value of the function at any input, while the covariance function (or kernel) defines how function values at different inputs are correlated.

Intuitively a GP calculates the mean function based on the observed data, and the covariance function defines how much we should expect function values to vary together based on their inputs. So points that are close together should have similar function values, while points that are far apart can have more different function values. How 'close' is defined is determined by the kernel choice and hyperparameters.

---

## [SLIDE 8 - GP Prior Samples]
**[2:50 - 3:05]**

Here we see samples from a GP prior. Each line represents a different plausible function before we've seen any data, but its key to remember that there is an infinite number of possible functions that could be drawn from this distribution, this is just a finite sample for visualization.
The kernel determines what kinds of functions are likely. 

On the left we have a standard RBF kernel, which produces smooth functions - and unless stated, this will be the default we use throughout the presentation.

On the right, we have a periodic kernel, which produces functions that repeat over regular intervals. 

As you can see, the choice of kernel has a huge impact on the kinds of functions the GP can represent. This is a key strength of GPs: we can encode our assumptions about the function we want to learn through the kernel choice. However, this also means that if we choose a poor kernel, our model may struggle to fit the data well. This motivates the need for more flexible models like Deep GPs that can learn complex kernels automatically.

---

## [SLIDE 9 - GP Posterior]
**[3:05 - 3:25]**

Using Bayes rule, we condition the GP prior across the observed data to give us the posterior distribution of functions. THis posterior mean function is our best estimate of the underlying function, while the posterior covariance quantifies our uncertainty about that estimate.

Notice how uncertainty—shown by the shaded region—is small near the data points where we have information, but grows larger far from the data. This uncertainty quantification is a key strengths of GPs. It gives us a single best guess, but also tells us how confident we should be in that guess. Many other machine learning models only give us a point estimate without any measure of uncertainty, which can be problematic in risk based contexts like credit risk modelling.

---

## [GP Summary]
**[3:25 - 3:50]**

To summarize: we get non-parametric flexibility, automatic uncertainty quantification, exact inference if the likelihood is Gaussian, and the ability to encode prior knowledge through kernel choice. However, they have significant limitations: cubic order scaling in the number of data points making them computationally expensive for large datasets. we have limited expressiveness, and GPs struggle with non-stationary functions. 

Kernel design is both a strength and weakness, allowing to encode assumptions but also requiring manual selection and tuning.

These limitations motivates finding a way to extend GPs to be more flexible and powerful, which brings us to Deep GPs.

---

## [Deep GPs: Motivation]
**[3:50 - 4:15]**

To address GP limitations, we need a more flexible model. The goal is to learn hierarchical representations of data, avoid manual kernel design, and capture complex non-stationary functions. The key idea, inspired by deep neural networks, is to stack GPs on top of each other—the output of one GP becomes the input to the next. Even when each individual layer uses a stationary kernel like RBF, the composition becomes non-stationary, allowing us to model much more complex functions.

---

## [DGP Definition]
**[4:15 - 4:45]**

Formally, a Deep GP with L layers is a hierarchical composition where each layer is a GP conditioned on the previous layer's output, each with its own kernel. The first layer takes our input X, produces output F-1, which feeds into layer 2, and so on, until we reach the final layer which produces our predictions with added observation noise. Each layer learns features from the previous layer, creating a deep hierarchy of learned representations.

---

## [Intractability]
**[4:45 - 5:05]**

However, there's a major challenge: This is the equation for the posterior distribution over the final layer's function values given the data. Notice that it involves integrating over all the intermediate layers' function values. This integral is intractable because each layer's output is a random variable that depends on the previous layer, creating a complex nested structure. So while DGPs are powerful in theory, we cannot directly compute the posterior on any realistic dataset. We need to use approximations.

---

## [Sparse Variational Inference]
**[5:30 - 6:00]**

The solution is variational inference. We can approximate the posterior with a simpler distribution q, where we assume q is a product of independent Gaussians for each layer. This is a mean-field approximation and it allows us to break the complex dependencies between layers. We then optimize q to make it as close as possible to the true posterior, which is equivalent to maximizing the Evidence Lower Bound (ELBO). Which we touch on in the next slide

Sparse variational inference introduces M, where M is far smaller than the number of data points N, psudo points at each layer. These inducing variables summarize the function at each layer, allowing us to parameterize the posterior through these M points rather than all N data points.

This mean field approximation does lose some uncertainty propagation between layers due to thee independence between layers assumption, but it allows us to train the model at all. 

This uncertainty propagation can be aproximately recovered through Monte Carlo sampling from the variational distribution at each layer and propagating those samples through the network, which gives us a more accurate estimate of the true posterior. This is called doubly stochastic variational inference, and is the method I use later in all my experiments, but due to time constraints I won't go into the details of that here, but it will be in the final dissertation.

---

## [DGP ELBO]
**[5:30 - 6:00]**

 The DGP ELBO has 2 terms, a data-fit term and a regularization term for each layer. Critically, this reduces complexity from order N-cubed to order N-M-squared, where M is typically between 100 and 1000, while N could be over 1 million. Giving huge savings in time.

---

## [DGP Summary]

To summarize Deep GPs: We use them to model complex non-stationary functions, learn hierarchical features that standard GPs cannot capture, and enable flexible kernel composition without manual design. The challenges are that exact inference is intractable requiring variational approximations, there's increased computational cost and hyperparameter tuning, non-convex optimization can get stuck in local minima.


---

## [Credit Risk Introduction]
**[6:25 - 6:45]**

Now let's see what these can actually do in practice. I've applied GPs and DGPs to credit risk modeling. The goal is to predict the probability of default for customers, which can inform intervention strategies like offering payment plans or credit adjustments to reduce losses from defaults.

SO we have a UCI Credit Card dataset,  30,000 transactions, 23 features, with either true or false labels for default. I trained a single-layer GP and two DGP architectures—2-layer and 3-layer—using doubly stochastic variational inference on a regression task of predicting a value 0-1 as a probability the customer defaults.

 We compared against a few standard ML models used for this.
 
  Models were evaluated using AUC-ROC, and Expected Calibration Error. These are very early results—full analysis and code will be in the final dissertation.

---

## [Results Comparison]
**[7:10 - 7:30]**

Here's the comprehensive comparison across all metrics. The key observation is that GPs and DGPs achieve very competitive discriminative performance—their AUC scores are comparable to the tree-based methods. However, the really striking difference shows up in calibration, which we'll discuss in more detail shortly.

---

## [SLIDE 22 - AUC-ROC Comparison]
**[7:30 - 8:15]**

In terms of AUC-ROC, all models perform similarly, with GPs and DGPs achieving competitive scores. 

This means that in terms of ranking customers by risk - putting a higher predicted probability for those who actually defaulted - GPs and DGPs do just as well as the best tree-based models.

However, AUC-ROC only tells us about the relative ordering of predictions, it only says the model can distinguish between defaulters and non-defaulters, it does not tell us how accurate the predicted probabilities are. 

It does not give usthe quality of the probability estimates themselves. This is where calibration comes in.

## [SLIDE 22 - Calibration Quality]
**[8:15 - 8:45]**

So with calibration The GPs and DGPs achieve near-perfect calibration with Expected Calibration Error below 0.011, meaning when they say there's a 30% chance of default, about 30% of those cases actually default.

The other tested models have ECE between 0.15 and 0.22—they're poorly calibrated. so a 30% predicted probability might only correspond to 10-15% actual defaults. This is a huge difference in the quality of probability estimates.
 
Also, on top of mean probability values, GPs and DGPs provide full uncertainty distributions over those probabilities. So not only do GPs and DGPs give a nearly perfect probability estimate, but they also tell us how confident we should be in that estimate, which standard ML models often do not, which can be an advantage in risk based decision making

---

## [SLIDE 23 - Key Takeaways]
**[8:45 - 9:20]**

To summerise: GPs and DGPs deliver similar discriminative performance to other machine learning models in terms of AUC scores.

 Second, there was no significant improvement from DGPs over standard GPs in this dataset—the added complexity didn't provide clear benefits. 
 
 Thirdly, calibration is massively better in GP based models. 1% calibration error verses 15-22% for tree models.

