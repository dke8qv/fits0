import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from math import log
from random import gauss

xmin=1.0
xmax=20.0
npoints=12       
sigma=0.2        
pars=[0.5,1.3,0.5]

lx=np.zeros(npoints)
ly=np.zeros(npoints)
ley=np.zeros(npoints)

def f(x,par):
    return par[0]+par[1]*log(x)+par[2]*log(x)*log(x)

def getX(x):
    step=(xmax-xmin)/npoints
    for i in range(npoints):
        x[i]=xmin+i*step
        
def getY(x,y,ey):
    for i in range(npoints):
        y[i]=f(x[i],pars)+gauss(0,sigma)
        ey[i]=sigma

# --- least squares fit ---
def fit_once():
    getX(lx)
    getY(lx,ly,ley)
    A = np.zeros((npoints,3))
    for i in range(npoints):
        L = log(lx[i])
        A[i,0] = 1
        A[i,1] = L
        A[i,2] = L*L
    
    W = np.diag(1.0/ley**2)
    AtW = A.T @ W
    cov = inv(AtW @ A)
    p = cov @ (AtW @ ly)
    
    chi2 = np.sum(((ly - A @ p)/ley)**2)
    ndf = npoints - 3
    return p, chi2, chi2/ndf

# --- run experiments ---
nexperiments = 2000

par_a = []
par_b = []
par_c = []
chi2_vals = []
chi2_red  = []

for _ in range(nexperiments):
    p, c2, c2r = fit_once()
    par_a.append(p[0])
    par_b.append(p[1])
    par_c.append(p[2])
    chi2_vals.append(c2)
    chi2_red.append(c2r)

par_a = np.array(par_a)
par_b = np.array(par_b)
par_c = np.array(par_c)
chi2_vals = np.array(chi2_vals)
chi2_red  = np.array(chi2_red)

# --- first 2x2: a, b, c, chi2 ---
fig1, ax1 = plt.subplots(2,2, figsize=(8,8))
ax1[0,0].hist(par_a, bins=40)
ax1[0,0].set_title("a distribution")
ax1[0,1].hist(par_b, bins=40)
ax1[0,1].set_title("b distribution")
ax1[1,0].hist(par_c, bins=40)
ax1[1,0].set_title("c distribution")
ax1[1,1].hist(chi2_vals, bins=40)
ax1[1,1].set_title("Chi2 distribution")
plt.tight_layout()

# --- second 2x2: parameter correlations + reduced chi2 ---
fig2, ax2 = plt.subplots(2,2, figsize=(8,8))
ax2[0,0].hist2d(par_a, par_b, bins=30)
ax2[0,0].set_title("b vs a")
ax2[0,1].hist2d(par_a, par_c, bins=30)
ax2[0,1].set_title("c vs a")
ax2[1,0].hist2d(par_b, par_c, bins=30)
ax2[1,0].set_title("c vs b")
ax2[1,1].hist(chi2_red, bins=40)
ax2[1,1].set_title("Reduced chi2")
plt.tight_layout()

# --- chi2 stats ---
ndf = npoints - 3
mean_chi2   = np.mean(chi2_vals)
std_chi2    = np.std(chi2_vals)
exp_mean    = ndf
exp_std     = np.sqrt(2*ndf)
mean_chi2_r = np.mean(chi2_red)

print("Mean chi2 =", mean_chi2, " Expected =", exp_mean)
print("Std chi2  =", std_chi2,  " Expected =", exp_std)
print("Mean reduced chi2 =", mean_chi2_r)

# --- generate one dataset again and plot fit ---
getX(lx)
getY(lx, ly, ley)

A = np.zeros((npoints,3))
for i in range(npoints):
    L = log(lx[i])
    A[i,0] = 1
    A[i,1] = L
    A[i,2] = L*L

W = np.diag(1.0/ley**2)
AtW = A.T @ W
cov = inv(AtW @ A)
p_fit = cov @ (AtW @ ly)

x_smooth = np.linspace(xmin, xmax, 200)
y_fit = p_fit[0] + p_fit[1]*np.log(x_smooth) + p_fit[2]*np.log(x_smooth)**2

# --- save all figures to PDF ---
with PdfPages('LSQFit.pdf') as pdf:
    pdf.savefig(fig1)
    pdf.savefig(fig2)

    # plot the fit with data
    plt.figure(figsize=(8,6))
    plt.errorbar(lx, ly, yerr=ley, fmt='o', label="Data with error bars")
    plt.plot(x_smooth, y_fit, label=f"Fit: a={p_fit[0]:.2f}, b={p_fit[1]:.2f}, c={p_fit[2]:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Least Squares Fit on One Dataset")
    plt.legend()
    pdf.savefig()
    plt.close()

    # comments page
    plt.figure()
    plt.axis('off')
    plt.text(0.01, 0.9, "Comments:", fontsize=14)
    plt.text(0.01, 0.8, f"• Mean χ² = {mean_chi2:.2f}  (Expected ≈ {exp_mean})")
    plt.text(0.01, 0.7, f"• Std χ²  = {std_chi2:.2f}  (Expected ≈ {exp_std:.2f})")
    plt.text(0.01, 0.6, f"• Mean reduced χ² = {mean_chi2_r:.2f} (Expected ≈ 1)")
    plt.text(0.01, 0.5, "• More data points → narrower parameter spreads.")
    plt.text(0.01, 0.4, "• Larger sigma → wider spreads and worse χ².")
    pdf.savefig()
    plt.close()

print("Saved LSQFit.pdf")
input("Press Enter to exit")


