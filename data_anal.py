import george
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from george import kernels
import scipy.optimize as op
freqs = ["21.7","11.2","7.7", "4.8", "2.3"]
results = dict()
coef = {
    "21.7": 7000,
    "11.2": 1000,
    "7.7": 1000,
    "4.8": 1000,
    "2.3": 7000
}
for freq in freqs:
    df = pd.read_csv("2253_1.06.18/2253.txt", sep='\t').filter(items=["#JD", freq,"sgm"+freq]).dropna()
    x = df["#JD"].as_matrix(columns=None)
    y = df[freq].as_matrix(columns=None)
    yerr = df["sgm"+freq].as_matrix(columns=None)
    kernel = np.var(y) * kernels.ExpSquaredKernel(metric=100, ndim=1, axes=0)
    gp = george.GP(kernel)
    gp.compute(x, yerr)    
    x_pred = np.linspace(x[0],x[-1], coef[freq]) 
    pred, pred_var = gp.predict(y, x_pred, return_var=True)
    print("Initial ln-likelihood: {0:2f}".format(gp.log_likelihood(y)))

    def neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.log_likelihood(y)

    def grad_neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y)

    result = op.minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
    gp.set_parameter_vector(result.x)
    print("\nFinal ln-likelihood: {0:.2f}".format(gp.log_likelihood(y)))
    pred, pred_var = gp.predict(y, x_pred, return_var=True)
    results[freq] = {
        "x": x,
        "y": y,
        "yerr": yerr,
        "x_pred": x_pred,
        "pred": pred,
        "pred_var": pred_var,
        "ln": gp.log_likelihood(y)
    }
print(results["21.7"]["pred"])
for i in range(0, len(freqs)): 
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.top'] = False
    mpl.rcParams['ytick.right'] = False
    mpl.rcParams['axes.linewidth'] = 4
    mpl.rcParams['font.size'] = 14
    plt.fill_between(results[freqs[i]]["x_pred"], results[freqs[i]]["pred"] - np.sqrt(results[freqs[i]]["pred_var"]), results[freqs[i]]["pred"] + np.sqrt(results[freqs[i]]["pred_var"]), color="k", alpha=0.2)

    plt.plot(results[freqs[i]]["x_pred"], results[freqs[i]]["pred"], "k", lw=1.5, alpha=.5, label=freqs[i] + " GHz")
    plt.errorbar(results[freqs[i]]["x"],results[freqs[i]]["y"],yerr=results[freqs[i]]["yerr"],fmt=".k",capsize=0)
    plt.grid(True, animated=True)
    plt.xlabel("JD")
    plt.ylabel("Flux density, Jy")
    plt.legend()
    plt.show()
