# Chapter 1 Exercises
### Exercise 1.1 *Self-Play*
**Suppose, instead of playing against a random opponent, the reinforcement learning algorithm described above played against itself, with both sides learning. What do you think would happen in this case? Would it learn a different policy for selecting moves?**

In this case, the algorithms would learn how to play against each other continuously. This means that their policy's would keep upgrading until there is no better policy. Their policies would converge to a best and at this point none of the machine-players would ever lose.

The scenario described above is similar to what two humans who play against other would do when learning how to play Tic-Tac-Toe. If one improves on the other iteratively, and the other does so too, they both become better until they both learn the best way of playing. This is only true if both keep improving continuously.

### Exercise 1.2 *Symmetries*
**Many tic-tac-toe positions appear different but are really the same because of symmetries, How might we amend the learning process described above to take advantage of this? In what ways would this change improve the learning process? Now think again. Suppose the opponent did not take advantage of symmetries. In that case, should we? Is it true then, that symmetrically equivalent positions should necessarily have the same value?**

Taking advantage of the symmetries would make the algorithm learn faster. Every group of symmetric states contains four states (one for every rotation of the board game), which can be reduced to a single one. This means that the space-state reduces with this approach, which makes it easier for the algorithm to learn the best policy since there are less states to learn the value of.

If the opponent did not take into account symmetries, it would mean that she is playing in a different way to what she would play if looking at the board in a different perspective. This implies that if the agent took into account symmetries while learning, it would not necessarily learn the best policy to beat the opponent in every symmetrically-similar game. Thus, equivalent possitions shouldn't necessarily have the same value.
