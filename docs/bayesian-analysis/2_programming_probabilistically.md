# 2 Programming Probabilistically

Find the bias of coin using PyMC3.

```python
with pm.Model() as coin_model:
    theta = pm.Beta("theta", alpha=1., beta=1.)
    y = pm.Bernoulli("y", p=theta, observed=data)
    trace = pm.sample(1_000, random_seed=42, chains=4)

az.plot_trace(trace)
```

`az.plot_trace(trace)` gives two subplots for each variable:

1. The Kernel Density Estimation
2. Individual sampled values at each step during sampling

`az.summary(trace)` returns a DataFrame with
mean, standard deviation, and 94% HPD interval.

`az.plot_posterior(trace)` shows a histogram or KDE for the posterior
along with the man and 94% HPD.

If we want to check an exact value—
such as a coin is fair and has a probability of 0.5—
you create a _Region Of Practical Equivalence (ROPE)_
such as $[0.45, 0.55]$.
Once defined:

1. If ROPE does not overlap with HPD, coin not fair
2. ROPE contains the entire HPD, coin is fair
3. ROPE partially overlaps, no conclusion

```python
az.plot_posterior(trace, rope=[0.45, 0.55], ref_val=0.5)
```

Can also use a loss function
that captures how different the true and estimated values are.

We usually do not have the true parameter to find the actual loss.
So we calculate the _expected loss function_—
the loss function averaged over the whole posterior.

The conjugate prior of the Gaussian mean is a Gaussian.

Almost every time when we measure the average of something using a big enough sample size
the average will be distributed as a Gaussian.

Your data does not always need to be literally Gaussian,
it can be an approximation.
If discrete variables are gathered in a Gaussianish manner,
you can still use the Gaussian distribution.

To access the values for any of the parameters in a `trace` object,
index it with the parameter name—`trace["var_name"]`.

Check the original data against the generated data using `pm.sample_posterior_predictive(trace, 100, model)`.
To plot the comparison:

```python
y_pred = pm.sample_posterior_predictive(trace, 100, model)
data_ppc = az.from_pymc3(trace=trace, posterior_predictive=y_pred)
az.plot_ppc(data_ppc, mean=False)
```

If outliers are messing with the fit,
one option is to drop outliers.
Other option is to choose different distribution.

Student's t-distribution is like a Gaussian,
but has an extra parameter which controls how normal-like it is—$\nu$.
There is no defined mean when $\nu < 1$,
and the variance of the distribution is only defined for values $\nu > 2$.
So be careful that the scale of the distribution is not the same as the standard deviation.

## Groups comparison

Focus on looking at effect size—
quantifying difference between two groups.

_Cohen's d_ is a way to measure effect size.
It measures the difference between means
with respect to the pooled standard deviation of both groups.
A Cohen's d of 0.5 could be interpreted as a difference of 0.5 standard deviation of one group
with respect to the other.

Can also look at probability that random point of data at one group is bigger than data in other group.
Compute the probability of superiority.

Can pass `shape` into distributions to create multiple priors.

```python
# The tips values
tip = tips["tip"].to_numpy()
# An array of the one hot encoded days of the week [0, 3, 2, 1, ...]
idx = pd.Categorical(
    tips["day"],
    categories=[
        "Thur",
        "Fri",
        "Sat",
        "Sun",
    ]
).codes
groups = len(np.unique(idx))

with pm.Model() as comparing_groups:
    mu = pm.Normal("mu", mu=0, sd=10, shape=groups)
    sigma = pm.HalfNormal("sigma", sd=10, shape=groups)

    y = pm.Normal("y", mu=mu[idx], sd=sigma[idx], observed=tip)

    trace_cg = pm.sample(5_000)
```

Then you can subtract the different posteriors from each other.

## Hierarchical models

In the case where you have multiple sub groups,
byt still want to leverage the information you get from the population,
you can use shared priors to estimate the parameters.

```python
with pm.Model() as model_h:
    mu = pm.beta("mu", 1., 1.)
    k = pm.HalfNormal("k", 10)

    theta = pm.Beta("theta", alpha=mu * k, beta=(1.0 - mu) * k, shape=len(N_samples))
    y = pm.Bernoulli("y", p=theta[group_idx], observed=data)
```
