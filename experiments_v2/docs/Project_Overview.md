# Risk-Aware Remaining Useful Life Prediction and Maintenance Decision Support with LinEx Loss and MC Dropout Project Overview

Our project's'motivation comes from mission-oriented fleet maintenance settings, where maintenance decisions depend on predicted component health, while the cost of prediction error is often asymmetric.So overestimating RUL can delay maintenance and increase failure risk, which makes standard symmetric regression objectives potentially misaligned with the actual decision problem. 

Our project is a controlled comparison built on the existing predictive maintenance. We keep the backbone model fixed as a CNN + Bi-LSTM time-series predictor and vary two elements in a structured way: the training loss and the uncertainty mechanism. 

- The loss comparison is between standard MSE $\mathcal{L}_{MSE} = (\hat{y} - y)^2$ and LinEx $\mathcal{L}_{LinEx}(e) = \exp(ae) - ae - 1$, where $e = \hat{y} - y$. LinEx is used to encode asymmetric error cost when optimistic RUL prediction is more dangerous. 
- The uncertainty comparison is between standard point prediction and Monte Carlo Dropout, which allows the model to output an approximate predictive distribution rather than only a single estimate. This gives us a clean four-run design for studying the separate and joint effects of asymmetric learning and predictive uncertainty.

Our goal is to answer three related questions. 

- 1. Does an asymmetric loss such as LinEx produce safer and more decision-relevant RUL predictions than a symmetric loss? 

- 2. Does predictive uncertainty from MC Dropout provide useful additional information for maintenance decisions? 

- 3. When the two are combined, does the gain come mainly from the training objective, from the uncertainty-aware inference stage, or from their interaction? This framing also reflects the professor’s feedback that the project should tell a clearer machine learning story than simply reusing an existing application pipeline.

In terms of experiments, the project begins with the NASA C-MAPSS setting and then extends to a synthetic degradation with controllable noise and uncertainty, especially under low-noise, medium-noise, and high-noise conditions.

Our project is intended to connect risk-sensitive machine learning with predictive maintenance decision support. Relative to the original maintenance papers, our emphasis is on whether the learning objective itself should reflect asymmetric operational risk, and whether uncertainty-aware prediction can make downstream maintenance planning more robust and informative.`
