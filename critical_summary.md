I forgot what the format of the summary is gonna be, so I am just going to write down my thoughts and revise this later

2/7/24 - Approximate Bayesian Computation
Summary - In the event that likelihood functions are a pain to calculate, we can bypass them using rejection sampling techniques.
By getting parameter estimates from our prior distribution and then generating data from each model, we can use a distance metric to see which models are closest to our actual data.
This generates an approximate distribution of our posterior distribution of the (i think) model parameter.

The ABC is sensitive to the distance metric used, but it also helps with computation time as typically statistics over the data are used, and the statistic is used in the computation.
The ABC is also sensitive to the statistic.
Other pitfalls include: how do we know the threshold will be good enough or just make computation intractable?
How do we know the statistics are actually good
Even with statistics the curse of dimensionality still exists
It's bayesian and has assumptions/oversimplistic priors (is this an issue, I definitely don't know).
