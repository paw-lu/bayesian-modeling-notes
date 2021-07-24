# 1 Thinking probabilistically

Bayesian modeling in three steps:

1. Given data and assumptions on how the data was generated,
   we design a model by combining probability distributions.
2. Condition the model on the data—
   use Bayes' theorem to add data to the model.
3. Check whether the model makes sense.

From a Bayesian perspective,
probability is measure that quantifies uncertainty.

$$
p(A, B) = P(A | B)p(B)
$$

Data is generated from some true probability distribution with unknown parameters.

Notation: $x$ is an instance from the objects random variable $X$.

Normal distribution

$$
x \sim \mathcal{N}(\mu, \sigma)
$$

$\sim$ is _distributed as_

Continuous variables can take any value in an interval.
Discrete can only take certain values.

Many models assume that values are sampled from the same distribution
and are independent of each other—
_independently and identically distributed (iid)_.

$$
P(\theta \mid y)=\frac{P(y \mid \theta) P(\theta)}{P(y)}
$$

We can use Bayes' theorem to tell us how to compute the probability of hypothesis $\theta$ given data $y$.

Terms:

- $p(\theta)$ **prior**
  reflects what we know about the value of $\theta$ before seeing the data
- $p(y | \theta)$ **likelihood**
  is how we introduce data.
  It is the plausibility of the data given the parameters.
- $p(\theta | y)$ **posterior**
  is the result and is given as a distribution.
- $p(y)$ **marginal likelihood**
  is the probability of observing the data averaged over all possible values.
  It is ignored since it is the denominator
  and we care about probabilities relative to each other.

Binomial distribution is the probability of getting a number of heads
given a number of coin tosses
and a probability of heads.

Beta is flexible and limited between 0–1
so it's good for probabilities.

The Beta distribution is the _conjugate prior_ of the Binomial.
A conjugate prior of a likelihood is a prior that—
when used in combination with a given likelihood—
returns a posterior with the same functional form as the prior.
When we use a Beta distribution as the prior for a Binomial likelihood
we get the Beta as a posterior distribution.
The Normal distribution is a conjugate with itself.

When reporting the distribution of parameters,
standard deviation works for normal-like distributions
but is misleading for others—
like skewed.
A _Highest-posterior density_ (_HPD_) interval
is the shortest interval containing a given portion of the probability density.
Most commonly used is 95% and 50%.

```python
# use ArviZ to plot that has mean and HPD
az.plot_posterior({'θ':stats.beta.rvs(5, 11, size=1000)})
```

To generate predictions

1. Sample $\theta$ from the posterior $p(\theta | y)$
2. Feed the value of $\theta$ to the likelihood $p(y | \theta)$

We can use this to make predictions
or to criticize our model by comparing the predictions to the real data.
This is _posterior predictive checks_.

