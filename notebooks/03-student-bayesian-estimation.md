---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.9.1
  kernelspec:
    display_name: bayesian
    language: python
    name: bayesian
---

# Estimation and Comparison with Probabilistic Programming

```python
import pymc3 as pm
```

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pymc3 as pm
from ipywidgets import interact
import arviz as az
import pandas as pd
import janitor
from utils import ECDF
import holoviews as hv
import hvplot.pandas

hv.extension("bokeh")


%load_ext autoreload
%autoreload 2
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

sns.set_style('white')
sns.set_context('talk')
```

## Learning Objectives of Part 3 onwards


1. Consolidate your knowledge of the Bayesian model building workflow and use probabilistic programming for parameter estimation;
2. Use probabilistic programming for hypothesis testing.


## Bayesian parameter estimation using PyMC3


Well done! You've learnt the basics of Bayesian model building. The steps are
1. To completely specify the model in terms of _probability distributions_. This includes specifying 
    - what the form of the sampling distribution of the data is _and_ 
    - what form describes our _uncertainty_ in the unknown parameters (This formulation is adapted from [Fonnesbeck's workshop](https://github.com/fonnesbeck/intro_stat_modeling_2017/blob/master/notebooks/2.%20Basic%20Bayesian%20Inference.ipynb) as Chris said it so well there).
2. Calculate the _posterior distribution_.

In the above, the form of the sampling distribution of the data was Binomial (described by the likelihood) and the uncertainty around the unknown parameter $p$ captured by the prior.


Now it is time to do the same using the **probabilistic programming language** PyMC3. There's _loads of cool stuff_ about PyMC3 and this paradigm, two of which are
- _probabililty distributions_ are first class citizens, in that we can assign them to variables and use them intuitively to mirror how we think about priors, likelihoods & posteriors.
- PyMC3 calculates the posterior for us: this is fancy math done for lazy programmers!

Under the hood, PyMC3 will compute the posterior using a sampling based approach called Markov Chain Monte Carlo (MCMC) or Variational Inference. Check the [PyMC3 docs](https://docs.pymc.io/) for more on these. 

From this notebook onwards, we have prepared a series of examples that will show you how to perform inference and prediction in a variety of problems. Hopefully, you will see that at the end of the day, everything we do boils down to estimation of some kind, and by doing it in a Bayesian setting, we avoid many pitfalls that come from blindly following canned statistical procedures.


## Example: Click-Through Rates


A common experiment in tech data science is to test a product change and see how it affects a metric that you're interested in. 

Say that we don't think enough people are clicking a button on my website, but we hypothesize that it's because the button is a similar color to the background of the page, meaning they're a difficult time finding the buttong to click.

We can serve up two pages, and randomly send some customers to each: the first the original page, the second a page that is identical, except that it has a button that is of higher contrast and see if more people click through. 

This is commonly referred to as an A/B test and the metric of interest is click-through rate (CTR), what proportion of people click through. We will use this example to help us build familiarity with PyMC3 mechanics.


### Load Data

Let's first load some data.

```python
ctr = (
    pd.read_csv('../data/clickthrough.csv', index_col=0)
    .label_encode('group')  # FYI: this is a pyjanitor function    
)
ctr.sample(10)
```

<!-- #region toc-hr-collapsed=false -->
### Build Model: Estimate $p$ for control group

Now it's time to build our probability model. Noticing that our model of having a constant click-through rate resulting in click or not is a biased coin flip:

- the sampling distribution is binomial and we need to encode this in the likelihood;
- there is a single parameter $p$ that we need to describe the uncertainty around. 
    - Its value must be bound between 0 and 1, so we must use a probability distribution that takes on these bounds.
    - Having not seen the data, we must express what we believe about the parameter $p$ -- this is nothing more than assigning credibility points to the number line for the value $p$. As a starter, we will show you how to express **"equal credibility from 0 to 1"**, by using the Uniform distribution.
<!-- #endregion -->

#### Model Definition

These are the ingredients for the model so let's now build it.

```python
control_df = ctr.query("group == 'control'")

with pm.Model() as model1_bernoulli:
    p = pm.Uniform("p", lower=0, upper=1)
    like = pm.Bernoulli("likelihood", p=p, observed=control_df["clicks"])
```

There is an alternative way to build this model, taking advantage of the fact that the sum of Bernoulli-distributed data follows a Binomial distribution. The syntax would look like this below, annotated with other modifications to guide you along:

```python
# Build model of p_a
with pm.Model() as model1_binomial:
    # Prior on p
    p = pm.Uniform("p")  # defaults for the uniform distribution are lower=0 and upper=1
    # Binomial Likelihood
    like = pm.Binomial(
        "likelihood",
        n=len(control_df),
        p=p,
        observed=len(control_df.query("clicks == 1")),
    )
```

Note how the data have to be in a different shape, though. With the Bernoulli likelihood, we need every single success/failure to be recorded. With the Binomial likelihood, we only need the summary statistics: number of trials, and number of successes.

<!-- #region -->
#### Sample from Posterior


It's now time to sample from the posterior using PyMC3. You'll also plot the posterior:
<!-- #endregion -->

```python
with model1_bernoulli:
    samples_bernoulli = pm.sample(2_000, tune=1_000)
```

```python
with model1_binomial:
    samples_binomial = pm.sample(2_000, tune=1_000)
```

#### Model Checking

Now, let's use ArviZ to perform some visual diagnostics.

```python
import arviz as az

# Posterior plot for bernoulli model
az.plot_posterior(samples_bernoulli, kind='hist');
```

```python
# Posterior plot for binomial model
az.plot_posterior(samples_binomial, kind='hist');
```

Notice how we get the same results using the two model formulations (Bernoulli- vs. binomial-distributed likelihoods).


#### Discussion

Interpret the posterior ditribution. What would your tell the non-technical manager of your growth team about the CTR?

<!-- #region toc-hr-collapsed=true -->
### Build Model: Compare $p$ for control and test groups

Having built the model for the control group, let's now extend it to compare the control and test groups.
<!-- #endregion -->

#### Hands-on: Build Model

Modify the first model (go ahead and copy/paste it here!) to estimate $p_{control}$ and $p_{test}$. Let's use Binomial likelihoods, for a bit of variety.

**Hint:** You will probably want to have two different `p`s (e.g. `p_control` and `p_test`) as well as two likelihoods (`like_control`, and `like_test`)

```python
# Create test_df, just as we created control_df above.
test_df = ctr.query("group == 'test'")

# Give your model a variable name
with pm.Model() as model2:
    # Copy p and likelihood for control group.
    p_control = pm.Uniform('p_control')
    like_control = pm.Binomial(
        'like_control', 
        n=len(control_df), 
        p=p_control, 
        observed=len(control_df.query("clicks == 1"))
    )
    
    # Modify the above p and likelihood for test group.
    p_test = pm.Uniform("p_test")
    like_test = pm.Binomial(
        "like_test",
        n=test_df.shape[0],
        p=p_test,
        observed=test_df.loc[lambda _df: _df.clicks ==1].shape[0]
    )
    

    # We will also explicitly compute the difference between
    # p_control and p_test. 
    # This shows you that we can do math on probability distributions!
    p_diff = pm.Deterministic("p_diff", p_test - p_control)
```

#### Sample from Posterior

```python
with model2:
    trace = pm.sample(2_000)
```

#### Model Checking

```python
# Which function?
# Which trace?
az.plot_posterior(trace, round_to=2);
```

#### Discussion

How did the test group compare to the control group?


#### Hypothesis Testing

Allen Downey wrote [a blog post](http://allendowney.blogspot.com/2016/06/there-is-still-only-one-test.html) that illustrates how all of statistical inference boils down into a single framework: test statistic, null model, and "probability of unexpectedness" (p-value) under that null model. All of the special statistical tests with fancy author names are merely particular cases of this framework.


#### Code-along: Translating probability distributions into things that matter

With Bayesian methods, we will go out on a limb to say that the p-value doesn't matter at all: posterior distributions are all we need, and we can use posterior distributions to do some cool stuff.

Let's make use of the posterior distribution on differences to calculate how much money we expect to gain.

If we know that our customers on average spend 25 USD after clicking, and 0 USD if they do not click, we can then simulate the distribution of expected increase of revenue over 1 million customers.

```python
x, y = ECDF(trace["p_diff"] * 25 * 1e6)
plt.plot(x, y)
plt.xlabel("Expected revenue increase")
plt.ylabel("Cumulative probability")
sns.despine()
```

Let's pretend that for each conversion,
we get \$25.
How much revenue do we get over a million customers?

```python
trace["p_diff"]
```

```python
dollar_dist = trace["p_diff"] * 25 * 1e6
x, y = ECDF(dollar_dist)
plt.plot(x, y)
plt.xlabel("Expected revenue increase")
plt.ylabel("Cumulative probabiliby")
sns.despine()
```

## Example: Baseball Players

We're now going to switch to a different dataset: that of baseball players and their batting stats. The goal of this analysis is to identify which player we want to target to make an offer to join our baseball team. 

To simplify the problem, we are going to make a decision on the basis of just batting average and their salaries.


### Learning Objectives

1. Reinforce Binomial/bernoulli generative story.
1. Illustrate how to use broadcasting to avoid writing for-loops. 
1. How to write a hierarchical model.


### Load Data

Let's load baseball player data.

```python
players = (
    pd.read_csv("../data/baseballdb/core/Batting.csv")
    .clean_names()
    .query("yearid == 2016")
    .select_columns(["playerid", "ab", "h"])
    .groupby("playerid")
    .sum()
)

salaries = (
    pd.read_csv("../data/baseballdb/core/Salaries.csv")
    .clean_names()
    .query("yearid == 2016")
    .select_columns(["playerid", "salary"])
    .groupby("playerid")
    .mean()
)

data = (
    players.join(salaries)
    .dropna()
    .reset_index()
    .label_encode("playerid")
    .set_index("playerid")
)
data
```

### Code-along: Build Model while avoiding for-Loops

Previously, we were able to compare two groups by copy/pasting code. That's not an ideal way to construct our model here. We're going to show you how you can avoid for-loops by taking advantage of broadcasting.


#### Model Definition

Once again, we see the Bernoulli distribution data generation story at play here: we have a number of trials (`ab`), with a number of successes (`h`), from which we want to estimate a player-specific property: `p`, the probability of hitting a pitch.

We are going to introduce a new distribution here: the `beta` distribution, which has the same desirable properties as the Uniform, but provides richer information.


Beta has parameters that let us control the shape,
like skewed left of right.
We can also specify a shape.

Since `p` has a shape,
then we are passing a vector of probability to `Beta`.

```python
# Give your model a name.
with pm.Model() as pitch_model:
    # we construct a vector of beta distributions, one for each player.
    p = pm.Beta("p", alpha=1, beta=1, shape=data.shape[0])
    like = pm.Binomial("like", p=p, n=data.ab, observed=data.h)
    
    # Let's also construct a "p per salary" metric, based on their salary.
    # We want highest p for lowest salary.
    pps = pm.Deterministic("pps", p / data.salary)

```

#### Sample from Posterior

```python
# Use the correct model variable here
with pitch_model:
    # Give your trace a name
    trace_batting = pm.sample(2_000)
```

#### Model Interpretation

Because we have 805 players, we're going to create a custom visualization that lets us scrub through our players' posterior distributions.

```python
from ipywidgets import SelectMultiple, HBox, VBox, Select, fixed

def dict2tuple(d):
    return list(*zip(*zip(d.items())))

def inversedict(d):
    return {v:k for k, v in d.items()}

playerid_mapping = data['playerid_enc'].to_dict()
playerid_inverse = inversedict(playerid_mapping)

player_select = SelectMultiple(
    options=dict2tuple(playerid_mapping),
    value=(0,)
)

metric_select = Select(
    options=['p', 'pps'],
)
```

```python
def scrub_posterior(player_encs: list, metric: str, trace):
    fig, ax = plt.subplots()
    for enc in player_encs:
        x, y = ECDF(trace[metric][:, enc])
        ax.plot(x, y, label=playerid_inverse[enc])
    ax.legend()
    ax.set_xlabel("p per salary")
    ax.set_ylabel("cumulative probability")

    sns.despine()


interact(
    scrub_posterior,
    player_encs=player_select,
    metric=metric_select,
    trace=fixed(trace_batting),
);
```

Interpreting ECDFs:

- **A straight line** mean equal credibility thoughout all probabilities
- **A backwards L shape** means value shifted to the right
- **A line with a downwards dip** means something in the middle

If we have one or 2 at bats,
we can get the unifrom distribution—
the straight line.
We can impose a stronger prior—
such as something that is much narrower by choosing different Beta parameters.

This is why Beta is useful.


#### Hands-on: Finding our ideal player

Using the visualization, find a bunch of players whom you think would be great to scout, based on their PPS.


### Hierarchical model

**Discussion:** Is it reasonable that some players have a `p` (probability of hitting) value that ranges from as small as 0 to as high as 1? Under what assumption would this be reasonable? Under what assumption would this be *unreasonable*?


#### Code-along: Model Definition

To build a hierarchical model, we essentially need to define a prior on the parameters of the Beta distribution. These new priors now become "hyperpriors". How do we pick hyperpriors then? Some qualitative rules of thumb:

1. Pick a distribution with the correct **support**. For example, if a parameter can/should never take negative values, then pick a distribution with only positive support.
1. The distribution should be **weakly informative**. For example, rather than a uniform distribution hyperprior, one might choose a wide distribution (relative to the relevant scale of the parameters) that spans 1 order of magnitude difference, but still has uneven allocation of credibility. 

Use the PyMC3 distribution library to your advantage! There are pictures that show you the shapes of the distributions, which can be helpful in narrowing your choices.


$\alpha$ and $\beta$ have distributions as well.
What's an appropriate distribution.
They covern the number of successes and failures for a player over the year.
So it must be positive and continuous.

Distributions that fit this are exponential,
half normal, gamma, log-normal, half gaussian.

Exponentials are easy
(one parameter)
though maybe not the best

```python
with pm.Model() as hierarchical_baseball:
    a_prior = pm.Exponential("a_prior", lam=1 / 29)
    b_prior = pm.Exponential("b_prior", lam=1 / 29)

    p = pm.Beta("p", alpha=a_prior, beta=b_prior, shape=data.shape[0])

    like = pm.Binomial("like", p=p, n=data["ab"], observed=data["h"])

    pps = pm.Deterministic("pps", p / data["salary"])
```

#### Sample from Posterior

```python
with hierarchical_baseball:
    trace_hierarchical = pm.sample(2_000)
```

While sampling happens, do you have any questions? Please feel free to ask.


#### Hands-on: Model Interpretation

```python
interact(
    scrub_posterior,
    player_encs=player_select,
    metric=metric_select,
    trace=fixed(trace_hierarchical),
);
```

**Discussion:** 

1. What do you notice about the posterior distributions on `p`?
1. What do you notice about the posterior distributions on `pps`?


We see more sigmoid distributed and a tighter range.
If justified,
a hierarchical model can help leverage what you know about the general population into specific instances.

Wild estimates being restriced to more realisitc intuitive values is known as **shrinkage**.

<!-- #region -->
## Model Building Workflow

**Things we did:**

1. Get data into correct shape.
1. Build naive model, followed by more complex model.
1. Build a simple loss function into our model.


**Things we didn't do:**

1. Posterior predictive checks.

**Bonus things:**

1. Build custom posterior explorers. You can use your favourite framework of choice.
<!-- #endregion -->

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

```python

```
