# Risk-Aware Remaining Useful Life Prediction and Maintenance Decision Support with LinEx Loss and MC Dropout Project Overview

## Team Project Summary

Our project studies whether risk-aware learning objectives can improve Remaining Useful Life prediction and downstream maintenance decisions under uncertainty. The central motivation is that in many predictive maintenance settings, overestimating RUL is more dangerous than underestimating it, because optimistic predictions can delay maintenance and increase the probability of failure.

Our current framework is built around a controlled 2×2 design on the NASA C-MAPSS benchmark:

- G1: MSE without MC Dropout
- G2: MSE with MC Dropout
- G3: LinEx without MC Dropout
- G4: LinEx with MC Dropout

This structure allows us to separately study the effect of:

1. replacing a symmetric regression loss with an asymmetric risk-aware loss
2. adding predictive uncertainty estimation through MC Dropout

## The difference between MSE and LinEx?

A key part of the project is that MSE and LinEx represent different assumptions about prediction error.

### MSE

Mean Squared Error treats positive and negative errors symmetrically:

$$\mathcal{L}_{MSE} = (\hat{y} - y)^2$$

This means that overestimation and underestimation of the same magnitude receive the same penalty. In standard regression settings this is often reasonable, but in predictive maintenance it can be problematic, because a large optimistic error is typically more costly than a similarly sized conservative error.

### LinEx

LinEx is an asymmetric loss designed to penalize one error direction more strongly than the other. In our setting, it is used to make dangerous RUL overestimation more expensive than underestimation.

A standard form is:

$$\mathcal{L}_{LinEx}(e) = \exp(ae) - ae - 1$$
where

$$e = \hat{y} - y$$

-  $a$ controls the asymmetry.

- When $a > 0$, positive errors are penalized more strongly than negative errors. In our project, this means that the model is encouraged to avoid optimistic RUL prediction when such optimism can lead to maintenance delay and failure risk.

So the main conceptual distinction is:

- MSE assumes symmetric error cost
- LinEx encodes asymmetric risk, which is more appropriate when one error direction is operationally more dangerous

This is why LinEx is not just a numerical tweak in our project. It changes the learning objective in a way that is directly aligned with the decision risk structure.

## What is MC Dropout doing in our project?

MC Dropout is not used here simply as regularization. Its main role in our project is to provide an approximate predictive distribution rather than only a single point estimate.

Instead of producing only one prediction, we keep dropout active at inference time and run the same input through the model multiple times. This produces a set of stochastic predictions:

From samples  $\hat{y}^{(1)}, \hat{y}^{(2)}, \dots, \hat{y}^{(T)}$ we compute:

- a predictive mean

$$\mu = \frac{1}{T}\sum_{t=1}^{T}\hat{y}^{(t)}$$

- and a predictive spread

$$\sigma^2 = \frac{1}{T}\sum_{t=1}^{T}\left(\hat{y}^{(t)} - \mu\right)^2$$

This is useful for our project because maintenance decisions should not rely only on a point prediction. Two engines can have the same predicted RUL but very different uncertainty levels. MC Dropout gives us a way to estimate that uncertainty and ask:

- when is the model unsure?
- does uncertainty correlate with real prediction difficulty?
- can uncertainty improve downstream maintenance decisions?

So in our project, LinEx shapes the training objective, while MC Dropout enriches the inference output.

Their roles are different:

- LinEx changes how the model learns from asymmetric error risk
- MCD changes what the model outputs at inference time, from a single point estimate to an approximate predictive distribution

This distinction is central to our research design.

## Current Model and Training Setup

The current backbone is a CNN + BiLSTM time-series model that takes sensor windows as input and predicts capped RUL. We keep the backbone fixed across groups so that the main comparisons isolate only:

- the loss function
- the uncertainty mechanism

This controlled design is important, because we want to avoid changing architecture and objective at the same time.

At the current stage, our experiments on FD001 suggest the following overall pattern:

- G4 performs best overall
- G3 also improves over the MSE baseline
- the main predictive gain appears to come from LinEx
- MC Dropout alone provides a weaker gain in point prediction, but still contributes useful uncertainty information

We have also implemented a reproducible multi-seed pipeline, protocol checks, structured outputs, and post-hoc analysis scripts.

## What We Have Already Analyzed

Beyond raw RMSE and NASA score, we now analyze the project in three layers.

### 1. Uncertainty validity

We tested whether MC Dropout uncertainty is actually informative. Our current results suggest that the model with the best point prediction is not necessarily the model with the best uncertainty-error coupling. This makes uncertainty quality a separate research question rather than a side effect of predictive accuracy.

### 2. Error structure

We studied where the gains come from across the RUL range. The current results suggest that the strongest overall group is not necessarily the strongest in the strict danger zone. A substantial part of the gain appears to come from safer global error structure and lower risk cost outside the most extreme region.

### 3. Maintenance decision layer

We also implemented a maintenance simulation layer. After an initial attempt using censored test trajectories, we revised the design and now use validation full trajectories reconstructed from saved split information. This allows more meaningful threshold-based maintenance analysis.

## What We Want to Test Next

The next stage is to move beyond one benchmark setting and test whether the observed gains reflect a more general mechanism.

### Round 7: Multi-dataset validation

We plan to extend the comparison from FD001 to FD003, FD002, and FD004 to test whether the current conclusions remain stable under different operating conditions and fault complexities.

### Round 8: Synthetic mechanism validation

We also plan to build a synthetic degradation benchmark with controllable noise and controllable uncertainty. This is especially important because we do not want the project contribution to rely only on one benchmark dataset.

The synthetic setting is intended to test whether:

- LinEx reduces dangerous RUL overestimation under noisy degradation signals
- MC Dropout provides useful predictive uncertainty for maintenance decisions
- the gain of LinEx + MC Dropout comes from training-stage bias shaping, inference-stage uncertainty-aware decision support, or both

In this setting, we want to explicitly control:

- noise intensity
- homogeneous vs heterogeneous uncertainty
- degradation shape
- maintenance capacity constraints

This would help us move from a benchmark-specific empirical result to a more general mechanism-level explanation.

## Main Research Direction

The broader question of the project is:

Can asymmetric risk-aware learning produce safer and more decision-relevant RUL forecasts than standard symmetric regression objectives, especially under uncertainty and maintenance constraints?

Our current evidence suggests that this is promising, but we want to validate it more carefully through cross-dataset experiments and controlled synthetic experiments.

## Expected Contribution

We expect the project to contribute in three ways:

1. a controlled empirical comparison of MSE, LinEx, and MC Dropout for RUL prediction
2. an analysis of how asymmetric bias and predictive uncertainty affect downstream maintenance decisions
3. a synthetic benchmark that helps explain when and why risk-aware learning is useful beyond a single real-world dataset
