# Problems

A list of the types of problems you can solve with probabilistic programming.

## 1

### Text-message

#### Problem

We have number of texts sent each day over a period of time.
Did the rate change?

#### Solution

This is a Poisson distribution,
and we need to know if the parameter changes.

Model this as two Poisson parameters that go into two Poison distributions.
The two Poisson parameters can be exponential distributions.
Poisson parameter hyperparameter can be the expected value identity and set it to the inverse of the mean.
We use a discrete uniform to model the day that the switch occurred bounded between the days.
Choose one of the poisson parameters depending on the value of the day switch and assign it to the final Poisson distribution.

See if the two parameters are different.
