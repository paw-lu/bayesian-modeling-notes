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

# Joint Probability, Conditional Probability and Bayes' Rule

```python
#Import packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set()
```

## Learning Objectives of Part 1-b


- To understand and be able to simulate joint probabilities and conditional probabilities;
- To understand Bayes' Theorem and its utility.


## Joint Probability & Conditional Probability


### Joint Probability


We have already encountered joint probabilities in the previous notebook, perhaps without knowing it: $P(A,B)$ is the probability two events $A$ and $B$ _both_ occurring.
* For example, getting two heads in a row.

If $A$ and $B$ are independent, then $P(A,B)=P(A)P(B)$ but be warned: this is not always (or often) the case.

One way to think of this is considering "AND" as multiplication: the probability of A **and** B is the probability of A **multiplied** by the probability of B.


#### Hands-On: Joint Probability and Coin Flipping


Verify that $P(A,B)=P(A)P(B)$ in the two fair coin-flip case (A=heads, B=heads) by 
- first simulating two coins being flipped together and calculating the proportion of occurences with two heads;
- then simulating one coin flip and calculating the proportion of heads and then doing that again and multiplying the two proportions.

Your two calculations should give "pretty close" results and not the same results due to the (in)accuracy of simulation. 

```python
# Solution: Calculate P(A,B)
rng = np.random.default_rng()
x_0 = rng.binomial(2, 0.5, 10_000)
p_ab = (2 == x_0).sum() / x_0.shape[0]

# Now, plot the histogram of the results
plt.hist(x_0);
print(p_ab)
```

```python
# Solution: Calculate P(A)P(B)
x_1 = rng.binomial(1, 0.5, 10_000)
x_2 = rng.binomial(1, 0.5, 10_000)
p_a = (1 == x_1).sum() / x_1.shape[0]
p_b = (1 == x_2).sum() / x_2.shape[0]
p_a * p_b
```

**Note:** In order to use such simulation and _hacker statistics_ approaches to "prove" results such as the above, we're gliding over several coupled and deep technicalities. This is in the interests of the pedagogical nature of this introduction. For the sake of completeness, we'll mention that we're essentially
- Using the proportion in our simulations as a proxy for the probability (which, although Frequentist, is useful to allow you to start getting your hands dirty with probability via simluation).

Having stated this, for ease of instruction, we'll continue to do so when thinking about joint & conditional probabilities of both simulated and real data. 


#### Hands-On: Joint probability for birds


What is the probability that two randomly selected birds have beak depths over 10 ?

```python
# Import data & store lengths in a pandas series
df_12 = pd.read_csv('../data/finch_beaks_2012.csv')
lengths = df_12['blength']

# Calculate P(A)P(B) of two birds having beak lengths > 10
p_a = (10 < lengths).sum() / lengths.shape[0]
p_b = p_a
p_a * p_b
```

* Calculate the joint probability using the resampling method, that is, by drawing random samples (with replacement) from the data. First calculate $P(A)P(B)$:

```python
# Calculate P(A)P(B) using resampling methods
n_samples = 100_000
p_a = (10 < lengths.sample(n_samples, replace=True)).sum() / n_samples
p_b = (10 < lengths.sample(n_samples, replace=True)).sum() / n_samples
p_a * p_b
```

Now calculate $P(A,B)$:

```python
# Calculate P(A, B) using resampling methods
n_samples = 100_000
samples_a = lengths.sample(n_samples, replace=True).reset_index(drop=True)
samples_b = lengths.sample(n_samples, replace=True).reset_index(drop=True)
p_ab = ((10 < samples_a) & (10 < samples_b)).sum() / n_samples
p_ab
```

**Task:** Interpret the results of your simulations.


### Conditional Probability


Now that we have a grasp on joint probabilities, lets consider conditional probabilities, that is, the probability of some $A$, knowing that some other $B$ is true. We use the notation $P(A|B)$ to denote this. For example, you can ask the question "What is the probability of a finch beak having depth $<10$, knowing that the finch is of species 'fortis'?"


#### Example: conditional probability for birds


1. What is the probability of a finch beak having depth > 10 ?
2. What if we know the finch is of species 'fortis'?
3. What if we know the finch is of species 'scandens'?

```python
# Q1 Answer
(10 < lengths).sum() / lengths.shape[0]
```

```python
# Q2 Answer
df_fortis = df_12.loc[df_12['species'] == 'fortis']
(10 < df_fortis.blength).sum() / df_fortis.shape[0]
```

```python
# Q3 Answer
df_scandens = df_12.loc[df_12['species'] == 'scandens']
(10 < df_scandens.blength).sum() / df_scandens.shape[0]
```

**Note:** These proportions are definitely different. We can't say much more currently but we'll soon see how to use hypothesis testing to see what else we can say about the differences between the species of finches.


### Joint and conditional probabilities

Conditional and joint probabilites are related by the following:
$$ P(A,B) = P(A|B)P(B)$$


**Homework exercise for the avid learner:** verify the above relationship using simulation/resampling techniques in one of the cases above.


![](../images/joint-conditional-marginal.png)


### Hands on example: drug testing


**Question:** Suppose that a test for using a particular drug is 99% sensitive and 99% specific. That is, the test will produce 99% true positive results for drug users and 99% true negative results for non-drug users. Suppose that 0.5% (5 in 1,000) of people are users of the drug. What is the probability that a randomly selected individual with a positive test is a drug user?

**If we can answer this, it will be really cool as it shows how we can move from knowing $P(+|user)$ to $P(user|+)$, a MVP for being able to move from $P(data|model)$ to $P(model|data)$.**


In the spirit of this workshop, it's now time to harness your computational power and the intuition of simulation to solve this drug testing example. 

* Before doing so, what do you think the answer to the question _"What is the probability that a randomly selected individual with a positive test is a drug user?"_ is? Write down your guess.

```python
# Take 10,000 subjects
n = 100_000
# Sample for number of users, non-users
users = rng.binomial(n, p=0.005)
non_users = n - users
```

```python
# How many of these users tested +ve ?
u_pos = rng.binomial(users, 0.99)
# How many of these non-users tested +ve ?
non_pos = rng.binomial(non_users, 0.01)
```

```python
# how many of those +ve tests were for users?
u_pos / (u_pos + non_pos)
```

**Discussion**: What you have been able to do here is to solve the following problem: you knew $P(+|user)=0.99$, but you were trying to figure out $P(user|+)$. Is the answer what you expected? If not, why not?

**Key note:** This is related to the serious scientific challenge posed at the beginning here: if you know the underlying parameters/model, you can figure out the distribution and the result, but often we have only the experimental result and we're trying to figure out the most appropriate model and parameters.

It is Bayes' Theorem that lets us move between these.


## 2. Bayes' Theorem

$$P(B|A) = \frac{P(A|B)P(B)}{P(A)}$$


As you may have guessed, it is Bayes' Theorem that will allow us to move back and forth between $P(data|model)$ and $P(model|data)$. As we have seen, $P(model|data)$ is usually what we're interested in as data scientists yet $P(data|model)$ is what we can easily compute, either by simulating our model or using analytic equations.


**One of the coolest things:** Bayes Theorem can be proved with a few lines of mathematics. Your instructor will do this on the chalk/white-board now.


### Bayes Theorem solves the above drug testing problem

Bayes Theorem can be used to analytically derive the solution to the 'drug testing' example above as follows.


From Bayes Theorem, 

$$P(user|+) = \frac{P(+|user)P(user)}{P(+)}$$




We can expand the denominator here into 

$$P(+)  = P(+,user) + P(+,non-user) $$

so that

$$ P(+)=P(+|user)P(user) + P(+|non-user)P(non-user)$$

and 

$$P(user|+) = \frac{P(+|user)P(user)}{P(+|user)P(user) + P(+|non-user)P(non-user)}$$.


Calculating this explicitly yields

$$P(user|+) = \frac{0.99\times 0.005}{0.99\times 0.005 + 0.01\times 0.995} = 0.332 $$


This means that if an individual tests positive, there is still only a 33.2% chance that they are a user! This is because the number of non-users is so high compared to the number of users.


Coming up: from Bayes Theorem to Bayesian Inference!
