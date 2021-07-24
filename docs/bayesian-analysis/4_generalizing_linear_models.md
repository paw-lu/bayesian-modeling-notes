# 4 Generalizing linear models

Use sigmoid for classification.

```python
with pm.Model() as model_0:
    alpha = pm.Normal("alpha", mu=0, sd=0)
    beta = pm.Normal("beta", mu=0, sd=0)
    mu = alpha + pm.math.dot(x_c, beta)
    theta = pm.Deterministic("theta", pm.math.sigmoid(mu))
    # The boundary decision — which determines the value used to
    # separate classes
    bd = pm.Deterministic("bd", - alpha / beta)
    y1 = pm.Bernoulli("y1", p=theta, observed=y_0)

    trace_0 = pm.sample(1_000)
```

