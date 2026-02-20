import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from typing import Optional

def plot_pr_curve(y_true, y_prob, label: Optional[str] = None):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.plot(rec, prec, label=label)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)