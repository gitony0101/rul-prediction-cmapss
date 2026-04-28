import torch
import numpy as np
from src_v2.rul.training.loss import LinExLoss
from src_v2.rul.evaluation.metrics import nasa_score

def test_linex_sign_convention():
    # If a > 0, overestimation (pred > target) should be penalized more than underestimation
    loss_fn = LinExLoss(a=1.0)
    
    target = torch.tensor([100.0])
    over_pred = torch.tensor([110.0])  # error = +10
    under_pred = torch.tensor([90.0])  # error = -10
    
    loss_over = loss_fn(over_pred, target)
    loss_under = loss_fn(under_pred, target)
    
    assert loss_over > loss_under, f"Overestimation loss {loss_over} should be > Underestimation loss {loss_under} for a > 0"

def test_error_convention():
    # e = y_pred - y_true
    # e > 0 means overestimation
    y_true = np.array([100.0])
    y_pred = np.array([110.0])
    e = y_pred - y_true
    assert e[0] > 0, "Error convention should be pred - true"

def test_nasa_score_sign():
    # nasa_score uses d = y_pred - y_true
    # overestimation (d > 0) is penalized by exp(d/10.0) - 1.0
    # underestimation (d < 0) is penalized by exp(-d/13.0) - 1.0
    y_true = np.array([100.0])
    over_pred = np.array([110.0])
    under_pred = np.array([90.0])
    
    score_over = nasa_score(y_true, over_pred)
    score_under = nasa_score(y_true, under_pred)
    
    # exp(10/10) - 1 = exp(1) - 1 approx 1.718
    # exp(10/13) - 1 approx 1.15
    assert score_over > score_under, "NASA score should penalize overestimation more for equal magnitude error"

if __name__ == "__main__":
    test_linex_sign_convention()
    test_error_convention()
    test_nasa_score_sign()
    print("All tests passed.")
