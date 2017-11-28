# Chapter 1 Exercises
### Exercise 1.1: *Self-Play*
**Suppose, instead of playing against a random opponent, the reinforcement learning algorithm described above played against itself, with both sides learning. What do you think would happen in this case? Would it learn a different policy for selecting moves?**

In this case, the algorithms would learn how to play against each other continuously. This means that their policy's would keep upgrading until there is no better policy. Their policies would converge to a best and at this point none of the machine-players would ever lose.

The scenario described above is similar to what two humans who play against other would do when learning how to play Tic-Tac-Toe. If one improves on the other iteratively, and the other does so too, they both become better until they both learn the best way of playing. This is only true if both keep improving continuously.

### Exercise 1.2: *Symmetries*
**Many tic-tac-toe positions appear different but are really the same because of symmetries, How might we amend the learning process described above to take advantage of this? In what ways would this change improve the learning process? Now think again. Suppose the opponent did not take advantage of symmetries. In that case, should we? Is it true then, that symmetrically equivalent positions should necessarily have the same value?**

Taking advantage of the symmetries would make the algorithm learn faster. Every group of symmetric states contains four states (one for every rotation of the board game), which can be reduced to a single one. This means that the space-state reduces with this approach, which makes it easier for the algorithm to learn the best policy since there are less states to learn the value of.

If the opponent did not take into account symmetries, it would mean that she is playing in a different way to what she would play if looking at the board in a different perspective. This implies that if the agent took into account symmetries while learning, it would not necessarily learn the best policy to beat the opponent in every symmetrically-similar game. Thus, equivalent possitions shouldn't necessarily have the same value.

### Exercise 1.3: *Greedy Play*
**Suppose the reinforcement learning player was *greedy*, that is, it always played the move that brought it to the position that it rated the best. Might it learn to play better, or worse, than a nongreedy player? What problems might occur**

If such a player learned to play, it might be the case that it would learn only local maxima policies. Since it would always select the best apparent next state, but hasn't learned the real expected value for each state, it might "think" that the policy that it has is the best possible one, never taking a risk.

In other words, the greedy algorithm might not necessarily perform better than others. Choosing the best option at each step might end up in no learning at all (since learning means approximating better the expected values for every state).

### Exercise 1.4: *Learning from Exploration*

**Suppose learning updates occurred after *all* moves, including exploratory moves. If the step-size parameter is appropriately reduced over time (but not the tendency to explore), then the state values would converge to a set of probabilities. What are the two sets of probabilities computed when we do, and when we do not, learn from exploratory moves? Assuming that we do continue to make exploratory moves, which set of probabilities might better to learn? Which would result in more wins?**

When we do not learn from exploratory moves, the value associated to each state corresponds to an estimate of selecting the greediest policy from that state onwards. In the limit, this estimate will converge to the real value for that state, and executing the greedy policy will return the best game.

Conversely, when we do learn from exploratory moves, the value associated to each state will not correspond to the estimate of selecting the greediest policy from that state onwards. Exploratory moves will either increase or decrease the state associated with the current state. If this value is decreased, the estimate for the current state will be updated and underestimated, possibly changing the current policy (which could be the optimal) for a different one. On the other hand, if the value of the current state is increased, it will become a closer estimate to its real value. Nevertheless, an approximation to the real state value estimates is already being done when non learning from exploratory moves (in the limit). This means that by updating the values (learning) when doing exploring, we're possibly underestimating the state values and this might imply obtaining a non-optimal policy.

Comparing both techniques, non-learning from exploratory moves will learn a better policy (set of probabilities), resulting in more wins, when compared to the policy and number of wins obtained by learning from exploratory moves.

### Exercise 1.5: *Other Improvements*

**Can you think fo other ways to improve the reinforcement learning player? Can you think of any better way to solve the tic-tac-toe problem as posed?**

*Not solved yet.*
