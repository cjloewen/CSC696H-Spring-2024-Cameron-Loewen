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



2/12/24 - Fast Epsilon-free inference of Simulation Models with Bayesian Conditional Density Estimation
After reading the paper closely, I still do not understand what Bayesian Conditional Density is.
I am assuming it has to do with proposition 1.
This paper was easier to read than the one due on wednesday (the one I have to present), and explained the math or at least gave a resource to understanding the math. However, some of the intution was difficult to follow as well as the presentation of the results/experiments. Proposition 1, which was the basis of the entire theory, should have been explained better and also what the downsides are.
It would seem that even though in the limit, this is true, in most practical applications, how close will it be, and how can we be assured that an MDN will get close enough to the true posterior.
In other words, with all this compounding uncertainty, how does it compare with just using an epsilon approximation in ABC.

The two algorithms also needed to be introduced better as well as their initialization better explained.

2/19/24 - Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
I thought the paper was simpler to read due to everything being pushed out to the appendix. That being said, I can't say I fully followed the math because of this decision. I also thought that all the dependencies being pushed away made it harder to follow and without much of the basics introduced I found it hard to understand what exactly they were doing and how it worked. I think the figures also could have been made easier to follow and understand.
Overall, I think I should have spent more time to read the paper and follow it better so I could at least ask questions on it.

2/21/24
