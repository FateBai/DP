# optimizers
## optimizer.py
### Example
```
>>> module = MyCustomModel()
>>> optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
>>> dp_optimizer = DPOptimizer(
...     optimizer=optimizer,
...     noise_multiplier=1.0,
...     max_grad_norm=1.0,
...     expected_batch_size=4,
... )
```
值得注意的是noise_multiplier是高斯分布的标准差的一部分，$\sigma = noise\_multiplier*max\_grad\_norm$，可以认为$\epsilon$和$\delta$蕴含在noise_multiplier之中。

```
optimizer: Optimizer,
noise_multiplier: float,
max_grad_norm: float,
expected_batch_size: Optional[int],
loss_reduction: str = "mean",
generator=None,
secure_mode: bool = False,
```

loss_reduction有'sum'和'mean'两种。
secure_mode表示是否应对浮点数攻击(floating point arithmetic attacks)。

### clip_and_accumulate()
这个函数主要进行梯度裁剪和聚合。
```python
per_sample_clip_factor = (
    self.max_grad_norm / (per_sample_norms + 1e-6)
    ).clamp(max=1.0)
```
这里考虑到了梯度范数可能是0的情况，进行了平滑处理（原论文没有提到）。
### add noise()
基本遵照*Deep Learning with Differential Privacy*，batch选择部分进行了变化。

![](/picture/2023-05-30-11-03-14.png)

## ddpoptimizer.py
DistributedDPOptimizer，考虑了分布式DP优化器。
