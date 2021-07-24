# Bayesian curve fitting

If we look at exponential decay,
there is an exponential decay with time parameter and offsets.

$$
y=A \cdot e^{-\frac{t}{\tau}}+C
$$

Where $A$ is the starting point,
$\tau$ is the half life,
and $C$ is the bias.

For $A$,
we should have something positive,
like $Exp(\lambda)$.

For $\tau$,
it must also be positive,
so we can use $Exp()\lambda$.

For likelihood,
can use normal since it will be reasonable.
It is an approximation of an approximation.
