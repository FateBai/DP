# accountants
[TOC]
## analysis/rdp.py
### Example
这个文件主要有两个功能，一个是计算RDP，另一个是计算最优的隐私损失。
```
Suppose that we have run an SGM applied to a function with L2-sensitivity of 1.

Its parameters are given as a list of tuples
``[(q_1, sigma_1, steps_1), ..., (q_k, sigma_k, steps_k)],``
and we wish to compute epsilon for a given target delta.

The example code would be:

>>> parameters = [(1e-5, 1.0, 10), (1e-4, 3.0, 4)]
>>> delta = 1e-5

>>> max_order = 32
>>> orders = range(2, max_order + 1)
>>> rdp = np.zeros_like(orders, dtype=float)
>>> for q, sigma, steps in parameters:
...     rdp += compute_rdp(q=q, noise_multiplier=sigma, steps=steps, orders=orders)

>>> epsilon, opt_order = get_privacy_spent(orders=orders, rdp=rdp, delta=1e-5)
>>> epsilon, opt_order  # doctest: +NUMBER
(0.336, 23)
```
首先看他给我们的例子，从parameter可以看出，batch有很多个，每个batch有一组对应参数。
**orders**实际上是$\alpha$的候选，允许**列表**或者**某个值**。
**compute_rdp**会计算每一轮的($\alpha$,$\rho$)-RDP的$\rho$值。
**get_privacy**主要干了两件事情，把($\alpha$,$\rho$)-RDP转成($\epsilon$,$delta$)-DP，选出最小的$\epsilon_{\alpha}$。
总结一下，rdp.py做了一件事情，使用RDP**跟踪隐私损失**。具体来说，计算每一轮RDP的损失，使用RDP线性组合的性质**累加**，最后转成DP选择最优的$\epsilon$和对应的$\alpha$输出。

### 如何计算($\alpha$,$\rho$)-RDP
```python
def _compute_log_a_for_int_alpha(q: float, sigma: float, alpha: int) -> float:
```
```python
def _compute_log_a_for_frac_alpha(q: float, sigma: float, alpha: int) -> float:
```

这两个函数是计算每一个batch隐私损失的核心函数。这两个函数的实现依据的是*Renyi Differential Privacy of the Sampled Gaussian Mechanism*这篇文章的SGM机制。
注意$\alpha$在这里被分类为**整数**和**小数**进行分别计算。论文告诉我们，SGM满足($\alpha$,$\rho$)-RDP, $\rho=\frac{1}{\alpha - 1}\ln{A_{\alpha}}$。而这两个函数就是在计算$\ln{A_{\alpha}}$
```python
def compute_rdp(
    *, q: float, noise_multiplier: float, steps: int, orders: Union[List[float], float]
) -> Union[List[float], float]:
```
compute_rdp返回$\rho$，注意这里$\rho=\rho_i*steps$是一个组合，我愿称之为同质组合，对于不同batch还需要进行异质组合。累加的理论依据如下。
![](/picture/2023-05-29-18-34-19.png)
###如何转化成($\epsilon$,$\delta$)-DP
```python
def get_privacy_spent(...)
eps = (
        rdp_vec
        - (np.log(delta) + np.log(orders_vec)) / (orders_vec - 1)
        + np.log((orders_vec - 1) / orders_vec)
    )
```
这里使用的是2019*Hypothesis Testing Interpretations and Rényi Differential Privacy*里的转换定理，相比17年*Rényi Differential Privacy*给出的转换更加tight。
二者对比如下（第一个是17年的结果，第二个是19年结果）。
![](/picture/2023-05-29-18-24-57.png)
![](/picture/2023-05-29-18-26-28.png)
```python
return eps[idx_opt], orders_vec[idx_opt]
```
如果是给定的$\alpha$，后者就没什么意义。

##Reference
*Mironov I, Talwar K, Zhang L. R\'enyi differential privacy of the sampled gaussian mechanism[J]. arXiv preprint arXiv:1908.10530, 2019.*

*Mironov I. Rényi differential privacy[C]//2017 IEEE 30th computer security foundations symposium (CSF). IEEE, 2017: 263-275.*

*Balle B, Barthe G, Gaboardi M, et al. Hypothesis testing interpretations and renyi differential privacy[C]//International Conference on Artificial Intelligence and Statistics. PMLR, 2020: 2496-2506.*

*Abadi M, Chu A, Goodfellow I, et al. Deep learning with differential privacy[C]//Proceedings of the 2016 ACM SIGSAC conference on computer and communications security. 2016: 308-318.*