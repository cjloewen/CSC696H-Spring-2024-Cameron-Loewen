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

2/14/24 - my presentation, don't have to write about this one.

2/19/24 - Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
I thought the paper was simpler to read due to everything being pushed out to the appendix. That being said, I can't say I fully followed the math because of this decision. I also thought that all the dependencies being pushed away made it harder to follow and without much of the basics introduced I found it hard to understand what exactly they were doing and how it worked. I think the figures also could have been made easier to follow and understand.
Overall, I think I should have spent more time to read the paper and follow it better so I could at least ask questions on it.

2/21/24 - Variational Dropout and the Local Reparameterization Trick
The paper extends on the idea of gaussian dropout. The paper explains that it generalizes this concept in order to be more flexible and have a bayesian inference approach. This allows the dropout weights to be learned rather than inferred or predetermined. Interestingly, there are really only empirical justifications for doing this rather than theory, so I am unsure of if this is sound, especially since they also varied the variational lower bound.

Most of the paper focused on being able to parallelize and simplify the computation both for speed and to limit the variance in the output. I did not really follow all the math, the presentation did help a lot. Overall, I thought the math manipulations for the efficiency of the computation were very impressive and made the approach very feasible even without full theoretical backing.

2/28/24 - Missed this Talk

3/1/24 - Infogan
I was unfamiliar with the GAN model prior to reading it, but the paper did a decent job explaining it at a thorough enough level to understand the point. The math was difficult to follow due to not being introduced to the GAN objective prior to reading. While I understood the objective and the importance of being able to disentangle features, I was unsure of how well the paper actually was able to disentangle or how easily it could.

The paper was able to optimize a mutual information between the generated output and latent variables based on the lower bound and had ties to bayesian inference. This was essentially being able to coerce the model into accepting different distributions as the natural distributions of independent features of the data. This is where the issue I have with the model comes in, how complicated do you have to make these distributions, and how many times do you have to train it or tinker with it until it spits out an acceptable model. Specifically, is choosing latent variable distributions and numbers become a hyperparameter, can it be learnable? - This kind of reminds me of unsupervised learning with k clusters, k is a hyperparameter.


3/11/24 - Information Dropout: Learning Optimal Representations Through Noisy Computation
I thought this paper's goal was a little unclear because of how many goals were presented. It appeared almost to be a synthesis of too many different papers.
What I mean is that it appeared to be the same as variational dropout at the endpoint yet with an added regularizer from another paper that was supposed to enforce to enforce disentanglement. Of course, the authors may have just been modest by saying their paper was a "small step" in exploring the relationship between information theory and deep learning.
I am not very familiar with information theory (hopefully will be soon), so some of the math was hard to get through, but I appreciated the way the paper was laid out and how clear their steps were in a logistic manner. However, I did find a run-on sentence that was half a paragraph long and I am surprised no one told them to break it up.

3/18/24 - Denoising Diffusion Probabilistic Models
  The flow of the paper was easy to follow and the mathematical motivations, while complex, were made clear and delineated in a neat way. The model itself seems very odd to me and there is no immediate intuitive reason to me why it would work. The model also seemed like it had many possible variations due to hyperparameters which were not explored in this paper. I understood that the mathematical derviations lead to "denoising score matching," but I was not entirely sure what that meant or how it was better (it seemed like it was only empirically justified to use as an objective/reparameterization).
  The paper also did not provide much in the way of easy comparisons with the performance of other models (at least not graphically), but overall the results were interesting. I would also like to know more about what they meant by "noncompetitive negative log likelihood score."


3/27/24 - 
I had questions on how exactly this method was implicit. Was it because the energy function being used implicitly satisfied other qualities without the need to account for them?
In the beginning of this paper, it said that one of the pros was the flexibility in the computation/quality trade-off, but I do not believe any of the results addressed this. Not much was said about the cost of sampling, just the parameterization.
I liked that they essentially only had to train one model that gave an approximation of the energy of a datapoint, but I was unable to follow how energy implicitly satisfied what they claimed it did. Further, I liked how they showed that the model could be used compositionally, which is cool in the sense that you can potentially train models as necessary for certain features or on certain data, and combine them as necessary.
I did not like that they showed how GAN has better inception scores but did not make it clear what EBM does better than GAN in a clear way. I believe they said GANs are not as good at inpainting or out-of-distribution detection, but the results on these topics were only against other likelihood-models.

4/1/24 - 
This paper was definitely different being that it had no experiments or results. It feels like rather than publishing a paper it could have been an article or a part of a textbook. The methods introduced, CD, SM, and NCE, all did not seem to be very efficient. SM and CD were more intuitive and the process made sense, but it seemed like SM was just intractable or relied on heavy assumptions and approximations. CD also is just shortened MCMC due to it taking too much computation. NCE did not make intuitive sense to me.
The one brief section on score based models was interesting, since the authors mentioned how efficient and effective they are but not much else.
