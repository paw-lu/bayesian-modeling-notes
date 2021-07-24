# 3 Linear regression

## Simple linear regression

$$
y \sim \mathcal{N}(\mu=\alpha+x \beta, \epsilon)
$$

```python
with pm.Model() as model_g:
    alpha = pm.Normal("alpha", mu=0, sd=10)
    beta = pm.Normal("beta", mu=0, sd=1)
    epsilon = pm.HalfCauchy("epsilon", 5)

    mu = pm.Deterministic("mu", alpha + beta * x)
    y_pred = pm.Normal("y_pred", mu=mu, sd=epsilon, observed=y)

    trace_g = pm.sample(2_000, tune=1_000)
```

`alpha` and `beta` will be correlated here,
which is problematic for some samplers.
Can remove correlation by centering `x` by subtracting the mean from each `x`.
Can also standardize data by also dividing it by standard deviation.

## Robust linear regression

To make the linear regression more robust to outliers,
use Student's T instead of Gaussian.

## Hierarchical linear regression

If we have $N$ related data groups—
each needing a regression—
we can use common hyperpriors between them.

```python
with pm.Model() as unpooled_model:
    alpha_tmp = pm.Normal("alpha_tmp", mu=0, sd=10, shape=M)
```

## Variable variance

If variance is not constant,
can also use linear motif to model it as well.

```python
with pm.Model() as model_vv:
    alpha = pm.Normal("alpha", sd=10)
    beta = pm.Normal("beta", sd=10)
    gamma = pm.HalfNormal("gamma", sd=10)
    sigma = pm.HalfNormal("sigma", sd=10)

    x_shared = shared(data.Month.values * 1.)

    mu = pm.Deterministic("mu", alpha + beta * x_shared ** 0.5)
    epsilon = pm.Deterministic("epsilon", gamma + sigma * x_shared)

    y_pred = pm.Normal("y_pred", mu=my, sd=epsilon, observed=data.Length)

    trave_vv = pm.sample(1_000, tune=1_000)
```

Can use `x_shared` to look up non-observed values:

```python
x_shared.set_value([0.5])
ppc = pm.sample_posterior_predictive(trave_vv, 2_000, model=model_vv)
```
