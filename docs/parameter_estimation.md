# Bayesian inference and parameter estimation

If we flip a biased coin several times
and want to estimate the probability of heads from the data

$$
P(p \mid D)=\frac{P(D \mid p) P(p)}{P(D)}
$$

Where $P(p)$ is the **prior** (distribution),
$P(D \mid p)$ is the **likelihood**,
and $P( p \mid D)$ is the **posterior**.

The prior is the distribution containing our knowledge about $p$ prior to the introduction of data $D$.
The posterior is the distribution containing our knowledge about $p$ after considering the data $D$.

Since you only care about the $P(p \mid D)$ values relative to each other,
you can usually omit the denominator—
$P(D)$.

What is the prior?
It is a probability between 0 and 1.
If we have not flipped any coins,
we don't know much else.
So all values of $p$ within the interval are equally likely.
$P(p) = 1$ for $0 \leq p \leq 1$.
This is an uninformative prior since it contains little information.

There are several types of priors:

- **Informative priors**
  express specific,
  definite information about a variable.
  Like if we got a perfect coin,
  we could choose a prior with a sharp peak at 0.5
- **Weakly informative prior**
  express partial information about a variable
  such as a peak at $p = 0.5$
  with a larger variance
- **Uninformative priors**
  Express no information about a variable,
  except what you know for sure,
  such as $0 \leq p \leq 1$.

The uniform distribution seems uninformative,
but you can think of it in terms of probability $p$
or something like odds ratio $r=\frac{p}{1-p}$.

Choosing an informed prior on $p$ accounts to choosing a non-uniform prior on $r$
and vice versa.
To avoid this from happening,
you can use **Jeffrey's prior**,
which is an uninformed prior that solves this problem.

$$
P(p)=\frac{1}{\sqrt{p(1-p)}}
$$
