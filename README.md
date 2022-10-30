# euro_option_greeks_malliavin

<h3 align="center">European Options Greeks using Malliavin Calculus</h3>

---

[Wissem Chabchoub](https://www.linkedin.com/in/wissem-chabchoub/) | [Contact us](mailto:chb.wissem@gmail.com)

## 📝 Table of Contents

- [📝 Table of Contents](#-table-of-contents)
- [🧐 About <a name = "about"></a>](#-about)
- [🎥 Repository Structure  <a name = "repo-struct"></a>](#-repository-structure)


## 🧐 About <a name = "about"></a>

In this notebook, we will different method to compute the price and the greeks of a european call option based on Monte Carlo simulation, differentiation and Malliavin Calculus.


1. Euro option MC pricing

```math
$$p(x)=e^{-rT}E[(X_T^x -K)_+]$$
```

```math
$$B_{t_{0}}=0, B_{t_{i}}=\sum_{j=1}^{i} \sqrt{t_{j}-t_{j-1}} G_{j} \text{ ; } i>0$$
```

2. Greeks

* Method 1 

```math
$$ \Delta(x) = \frac{p(x+h)-p(x-h)}{2h} $$ 
```
```math
$$ \Gamma(x) = \frac{p(x+h)+p(x-h)-2P(x)}{h^2} $$ 
```

* Method 2 

```math
$$\Delta(x)=e^{-r T} E\left[\frac{B_{T}}{x \sigma T} F\right]$$
```
```math
$$ \Gamma(x)=e^{-r T} E\left[\left(\frac{-B_{T}}{x^{2} \sigma T}+\frac{\left(B_{T}\right)^{2}-T}{(\sigma T x)^{2}}\right) F\right]$$ 
```

* Method 3 

```math
$$
\begin{array}{c}
H_{\delta}(s)=1_{] K+\delta,+\infty[}(s)+\frac{s-(K-\delta)}{2 \delta} 1_{[K-\delta, K+\delta]}(s) \\
G_{\delta}(t)=\int_{-\infty}^{t} H_{\delta}(s) d s \\
F_{\delta}(t)=(t-K)_{+}-G_{\delta}(t)
\end{array}
$$
```
```math
$$
\Delta(x)=\frac{e^{-r T}}{x} E\left[H_{\delta}\left(X_{T}^{x}\right) X_{T}^{x}\right]+e^{-r T} E\left[\frac{B_{T}}{x \sigma T} F_{\delta}\left(X_{T}^{x}\right)\right]
$$
```

```math
$$
\begin{array}{c}
I_{\delta}(s)=\frac{1}{2 \delta} 1_{[K-\delta, K+\delta]}(s) \\
F_{\delta}(t)=(t-K)_{+}-\int_{-\infty}^{t} \int_{-\infty}^{s} I_{\delta}(u) d u d s
\end{array}
$$
```

```math
$$
\Gamma(x)=\frac{e^{-r T}}{x^{2}} E\left[I_{\delta}\left(X_{T}^{x}\right)\left(X_{T}^{x}\right)^{2}\right]+e^{-r T} E\left[\left(\frac{-B_{T}}{x^{2} \sigma T}+\frac{\left(B_{T}\right)^{2}-T}{(\sigma T x)^{2}}\right) F_{\delta}\left(X_{T}^{x}\right)\right]
$$
```


## 🎥 Repository Structure  <a name = "repo-struct"></a>


1. `euro_option_greeks_malliavin.ipynb`: A Jupyter Notebook 
2. `src` : Source code folder
3. `requirements.txt`: Requirements files
