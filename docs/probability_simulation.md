# Probability simulation

## Summary

| Name        | Abbreviation | Shape                               | Support (domain) | Story                                                                      |
| ----------- | ------------ | ----------------------------------- | ---------------- | -------------------------------------------------------------------------- |
| Uniform     | U(a, b)      | Box                                 | a < x < b        | Equal credibility assigned across (a, b) interval                          |
| Bernoulli   | Bern(p)      | Frequency of fail and success       | [0, 1]           | Trial that has 2 outcomes with probability $p$ for the success             |
| Poisson     | Pois(λ)      | Frequency count                     | int(x > 0)       | _Rare_ events occurring with rate $λ$ per unit time.                       |
| Exponential | Expon(β)     | Exponential/logrithmic growth/decay | x > 0            | Waiting time between Poisson process events                                |
| Beta        | Beta(ɑ, β)   |                                     | (0, 1)           | Expected fraction of successes out of the number of successes and failures |

## Uniform distribution

All events equally likely.

### Mental model

To estimate how many users will click through with a CTR of 0.7,
create a uniform distribution and count the number of users with a value greater than or equal to 0.7.

## Binomial distribution

The number of heads you get when flipping a coin $n$ times with probability $p$ of heads.
Also known as Bernoulli trials.

### Mental model

Process has a binary outcome (heads or not, click or not, etc)
and one of the two events occurs with probability $p$.

## Poisson distribution

The number of events that will occur in an interval of time where each event is
completely independent of the previous event.

### Mental model

You are waiting at the bus stop for a ride.
The amount of time you have to wait between buses is completely independent.
It could be 3 seconds, or 20 minutes.
Need the average amount of buses per hour to model the number of buses per hour.

Examples are:

- Number of births in a hospital
- Landings on a website
- Meteor strikes
- Aviation incidents

## Exponential distribution

### Mental model

When waiting for a bus,
the waiting time between buses.
The mean waiting time is the distribution of times between buses.

## Normal distribution

Any quantity that emerges as the sum of a large number of subprocesses tends to be Normally distributed
provided none of the subprocesses is very broadly distributed.

Generated given a mean and standard deviation.
