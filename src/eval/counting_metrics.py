import numpy as np

def mae(pred, gt):
    return float(np.abs(np.asarray(pred, float) - np.asarray(gt, float)).mean())

def wape(pred, gt):
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    return float(np.abs(pred - gt).sum() / gt.sum()) if gt.sum() > 0 else float("nan")

def sce(pred, gt):
    return (np.asarray(pred, float) - np.asarray(gt, float)).tolist()
