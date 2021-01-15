# 1 Introduction

## Bayesian inference

_Frequentist_ statistics assume that probability is the long-run frequency of events.
This becomes difficult when an even has no long-term frequency—
like presidential elections.
Bayesian statistics interpret a probability as a measure of belief.

Our belief about event $A$ is written as $P(A)$—
the _prior probability_.
An updated belief given evidence $X$ is $P(A|B)$,
the _posterior probability_.

Where frequentist methods return a number representing an estimate,
Bayesian methods return probabilities of different values.

As more data is acquired the prior belief is washed out by evidence.
As the number of instances $N$ goes to $\inf$,
Bayesian results often align with frequentist results.
At small $N$,
there is more instability,
and Bayesian method's prior and probabilities reflect that.

$$
P(A \mid X)=\frac{P(X \mid A) P(A)}{P(X)} \\
P(A \mid X) \propto P(X \mid A) P(A)
$$

As $N$ gets larger,
the distribution of posterior probability values gets sharper.
The uncertainty is proportional to the width of the curve.

### Bug or no bug

Let $A$ be an event that our code has no bugs.
Let $X$ be the even that the code passes all tests.
The prior will be set as a variable $P(A) = p$.
We want $P(A | X)$.

$$
P(X | A) = 1 \\
P(X) = P(X \text{ and } A) + (X \text{ and } \sim A) \\
= P(X | A)P(A) + P(X | \sim A)P(\sim A) \\
= P(X | A)p + P(X | \sim A)
$$

$P(X | \sim A)$ is subjective—
the code can pass and still have bugs.
We can estimate $P(X | \sim A) = 0.5$.
Then

$$
\begin{aligned}
P(A \mid X) &=\frac{1 \cdot p}{1 \cdot p+0.5(1-p)} \\
&=\frac{2 p}{1+p}
\end{aligned}
$$

We can plot out the posterior probability as a function of prior values between 0 and 1
($p \in [0,1]$).

Since $P(A | X )$ is the probability that there is no bug given we saw all tests pass,
$1 - P(A | X )$ is the probability that there is a bug given all tests pass.

## Probability distributions

If $Z$ is a random variable.
Then associated with $Z$ is a probability distribution function that assigns probabilities to the different outcomes $Z$ can take.

$Z$ can be:

1. Discrete. Only assume specific values.
2. Continuous. Any exact value.
3. Mixed. Both continuous and discrete.

### Discrete

When discrete $Z$'s distribution is called a _probability mass function_
and measures the probability $Z$ takes on the value $k$—
$P(Z = k)$.

One is the _Poisson_ distribution.

$$
P(Z=k)=\frac{\lambda^{k} e^{-\lambda}}{k !}, \quad k=0,1,2, \ldots
$$

Where $\lambda$ is called the parameter of the distribution.
It controls the shape.
For Poisson,
$\lambda$ is any positive number.
Higher $\lambda$ means more probability to larger values.
It is the _intensity_ of the Poisson distribution.

$k$ is a non-negative integer.

It can be written as

$$
Z \sim \text{Poi}(\lambda)
$$

In the Poisson distribution the expected value is equal to the parameter:

$$
E\large[ \;Z\; | \; \lambda \;\large] = \lambda
$$

## Continuous

When continuous,
the distribution is a _probability density function_.
One such distribution is a random variable with _exponential density_.

$$
f_Z(z | \lambda) = \lambda e^{-\lambda z }, \;\; z\ge 0
$$

An exponential random variable can only take on non-negative values,
but they don't have to be integers.

Can be written as

$$
Z \sim \text{Exp}(\lambda)
$$

$$
E[\; Z \;|\; \lambda \;] = \frac{1}{\lambda}
$$

## What is $\lambda$?

$\lambda$ isn't real.
We can talk about what $\lambda$ is likely to be by assigning a probability distribution to it.

