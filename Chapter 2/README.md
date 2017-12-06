# Chapter 2 Exercises
### Exercise 2.1
**In ε-greedy action selection, for the case of two actions and ε = 0.5, what is the probability that the greedy action is selected?**

The probability is 0.75%, which is broken down in the following way:
- 0.5% probability derived of choosing the greedy action, which is 1-ε = 1-0.5 = 0.5
- 0.25% chance derived of being chosen as the random option = 0.5 x 0,5.


### Exercise 2.2: *Bandit Example*
**Consider a *k*-armed bandit problem with *k* = 4 actions, denoted as 1, 2, 3, and 4. Consider applying to this problem a bandit algorithm using ε-greedy action selection, sample-average action-value estimates, and initial estimates of Q₁(*a*) = 0, for all *a*. Suppose the initial sequence of actions and rewards is A₁ = 1, R₁ = 1, A₂ = 2, R₂ = 1, A₃ = 2, R₃ = 2, A₄ = 2, R₄ = 2, A₅ = 3, R₅ = 0. On some of these time steps the ε case may have occurred, causing an action to be selected at random. On which time steps did this definitely occur? On which time steps could this possibly have occurred?**

The sequence of steps associated to the certainty of ε having occurred is shown next:
1. Maybe: Q(*a<sub>k<sub>*)=0 ∀k at this point, so choosing A₁ = 1 could have happened as a random choice due to ε, or by randomly choosing an action when tie-breaking.
2. Yes: Q(*a<sub>1<sub>*)>0 at this point, so choosing A<sub>2</sub> = 2 is necessarily a random decision, otherwise A<sub>1</sub> would have been chosen.
3. Maybe: Q(*a<sub>k<sub>*) = 1 for k ∈ {1, 2} at this point, so choosing A<sub>3</sub> = 2 could have happened due to ε, or by randomly choosing an action when tie-breaking.
4. Maybe: Q(*a<sub>2<sub>*) = 3 at this point, so choosing A<sub>4</sub> = 2 could have happened due to ε, or by selecting the option action with highest Q at this point, which is a<sub>2</sub>.
5. Yes: Q(*a<sub>2<sub>*) = 5 at this point, so choosing A<sub>5</sub> = 3 A<sub>2</sub> = 2 is necessarily a random decision, otherwise A<sub>2</sub> would have been chosen.

### Exercise 2.3:
**In the comparison shown in Figure 2.2, which method will perform best in the long run in terms of cumupative reward and probability of selecting the best acton? How much better will it be? Express your answer quantitatively.**

The method that will perform best in the long run is ε-greedy with ε = 0.01. Though it seems like the ε-greedy method with ε = 0.1 works better for 1000 steps, in the long run ε = 0.01 should outperform it. This is true because in the limit when steps tend to infinity, both methods will have the expected rewards converge to the true reward value. Since ε = 0.01 has a lower degree of randomness, it will capture a higher amount of reward. In either case any method will have an infinite reward, though the area under the curve will be bigger for ε = 0.01 if we were to select a given step much bigger than 1000 and evaluate the curve until that timestamp.

Quantitatively, we can estimate the reward obtained in each step by calculating the expected reward knowing q*(a) (the real value for each action). The expected reward value of a random guess is approximately 0, which means that a random guess' expected reward is 0. For ε = 0.1, there's a 90% chance of having a 1.5 reward value, and a .1% of having a random guess, which means a 0 reward. This results in an expected reward of 1.5 * .9 = 1.35.

We can estimate the reward for ε = 0.01 in a similar way. For ε = 0.01, there's a 99% chance of having a 1.5 reward value, and a .01% of having a random guess with 0 reward. This results in an expected reward of 1.5 * .99 = 1.485. Thus, ε-greedy with ε = 0.01 results in an expected reward value which is 0.135 units higher than ε-greedy with ε = 0.1.
