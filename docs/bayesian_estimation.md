# 3 Bayesian estimation

Basics of Bayesian modeling are

1. Completely specify the model in terms of probability distributions.

   - Specify the form of the sampling distribution of the data
   - What form describes our uncertainty in the unknown parameters

2. Calculate the posterior distribution.

PyMC3 will calculate the posterior for us.
It will use a sampling based approach—
Markov Chain Monte Carlo (MCMC)
or Variational Inference.
[PyMC docs](https://docs.pymc.io/)

If you know nothing about probability $p$,
can assign it to Uniform distribution between 0 and 1

```py
pm.Uniform("p", lower=0, upper=1)
```

If you have a coin flip problem,
can model it using Bernoulli and counting the events,
or Binomial and counting the sum of events.

Can find the posterior distribution for a test and control,
and find their different to see the distribution of change.

If you were interested in the probability,
you could not do a t-test for this,
since probability is bounded and not normally distributed.

Instead of using the uniform distribution for an uniformed prior,
can use Beta distribution.
Still has desirable properties of uniform distribution,
but with extra parameters that let us control the shape and skew direction.

Sometimes you can get unrealistic posterior distributions,
like equal probability from 0-1 on batting averages.
If we impose a more opinionated prior we can modify some of these distributions,
this is why Beta is useful—
it has parameters you can use to change the shape.

You can also define a prior on the parameters of the Beta distribution,
which itself is the prior $p$.

You need to pick a prior with the correct support (domain),
and it should be weakly informative.
For example rather than a uniform distribution,
you can choose a wide distribution that spans 1 order of magnitude difference,
but still has an uneven allocation of credibility.

Use the PyMC3 distribution library!
Docs have illustrations of shapes.

For probability of player hitting,
need something positive and continuous.

Estimates having their domain restricted by choosing more opinionated priors is known as shrinkage.

When choosing a distribution:

1. Find something with the correct support (domain)
2. Think about likelihood function.
   How is the data distributed.
   Amount happening in time?
   Poisson.
   Positive negative?
   Bernoulli/Binomial.
   Unsure?
   Normal.
   If other processes?
   Negative binomial is number of failures until a success.
   Zero inflated distributions.
   Zero inflated Poisson distribution is two disributions—
   one that generates a log of zeros and one that generates the Poisson side of it.
3. The shape of the distribution.
   Look at PDF.
   What is the domain,
   skew,
   what is in centered on,
   how tight in the distribution?
