# 2 More PyMC3

## Model

In PyMC3 all variables are handles within the context of the `Model` context.

```python
import pymc3 as pm

with pm.Model() as model:
    parameter = pm.Exponential("poisson_param", 1.0)
    data_generator = pm.Poisson("data_generator", parameter)
```

Any variables created in the context of `Model` will be automatically assigned to that model.

We can continue to work within the context of the same model by using `with` and the name assigned before.

```python
with model:
    data_plus_one = data_generator + 1
```

We can examine the same variables outside of the model context once defined.
But need to be within a context to define more.

```python
parameter.tag.test_value
```

Each variable assigned to a model will be defined with its own name—
the first string parameter.
To create a different model object with the same name as the one we used previously:

```python
with pm.Model() as model:
    theta = pm.Exponential("theta", 2.0)
    data_generator = pm.Poisson("data_generator", theta)
```

If we want to define another separate model,
use another name:

```python
with pm.Model() as ab_testing:
    p_A = pm.Uniform("P(A)", 0, 1)
    p_B = pm.Uniform("P(B)", 0, 1)
```

PyMC3 gives you notifications about transformations when you add variables.
This is done internally to modify the space the variable is sampled in.

## Variables

All variables have an initial (test) value.

```python
parameter.tag.test_value
data_generator.tag.test_value
data_plus_one.tag.test_value
```

`test_value` is used only for the model as a starting point for sampling
if no other start is specified.
The initial state can be changed during creating using the `testval` param.

```python
>>> with pm.Model() as model:
>>>   parameter = pm.Exponential("poisson_param", 1.0, testval=0.5)
>>> parameter.tag.test_value
0.499
```

This can be helpful if you are using an unstabler prior that requires a better starting point.

PyMC3 has two variables:

1. _stochastic variables_ are non-deterministic—
   even if you knew the values of all parameters and components
   it would still be random.
   Like `Poisson`, `DiscreteUniform`, and `Exponential`.
2. _deterministic variables_
   are not random if the parameters and components are known.

### Initializing stochastic variables

Stochastic variables require a `name` argument
plus the additional class-specific parameters.

```python
some_variable = pm.DiscreteUniform("discrete_uni_var", 0, 4)
```

The `name` attribute is used to retrieve the posterior distribution later in the analysis.

Gor multivariable problems,
instead of making an array of stochastic variables
use the `shape` parameter to create a multivariate array of independent stochastic variables.

```python
# Instead of this
beta_1 = pm.Uniform("beta_1", 0, 1)
beta_2 = pm.Uniform("beta_2", 0, 1)
...

# Do this

betas = pm.Uniform("betas", 0, 1, shape=4)
```

### Deterministic variables

```python
deterministic_variable = pm.Deterministic(
  "deterministic variable", some_function_of_variables
)
```

`Deterministic` is not the only way to create deterministic variables.
Elementary options—
addition, exponential, etc—
implicitly create deterministic variables.

```python
with pm.Model() as model:
    lambda_1 = pm.Exponential("lambda_1", 1.0)
    lambda_2 = pm.Exponential("lambda_2", 1.0)
    tau = pm.DiscreteUniform("tau", lower=0, upper=10)

new_deterministic_variable = lambda_1 + lambda_2
```

If we want a deterministic variable to be tracked by our sampling though,
we need to explicitly define it as a named deterministic variable with the constructor.

From the last chapter,
we had a case where

$$
\lambda =
\begin{cases}
\lambda_1  & \text{if } t \lt \tau \cr
\lambda_2 & \text{if } t \ge \tau
\end{cases}
$$

```python
import numpy as np

n_data_points = 5
idx = np.arrange(n_data_points)
with model:
    lambda_ = pm.math.switch(tau >= idx, lambda_1, lambda_2)
```

`switch` here is deterministic.

Inside a deterministic variable,
the stochastic variables behave like scalars
(or NumPy arrays if multivariable).
We can do whatever as long as dimensions match up in the calculations.

```python
def subtract(x, y):
    return x - y

stochastic_1 = pm.Uniform("U_1", 0, 1)
stochastic_2 = pm.Uniform("U_2", 0, 1)

det_1 = pm.Deterministic("Delta", subtract(stochastic_1, stochastic_2))
```

Expressions we are making must be compatible with Theano tensors.
