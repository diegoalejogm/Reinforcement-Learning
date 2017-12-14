# Chapter 4 Exercises
### Exercise 4.1
**In Example 4.1, if π is the equiprobable random policy, what is q<sub>π</sub> (11, down)? What is q<sub>π</sub> (7, down)?**

- q<sub>π</sub> (11, down) = -1 since E[ R<sub>t+1</sub> | 11, down ]= -1 and E[ G<sub>t+1</sub>|s ] = 0
- q<sub>π</sub> (7, down) = -15 since E[ R<sub>t+1</sub> | 7, down ]= -1 and E[ G<sub>t+1</sub>|s ] = -14

### Exercise 4.1
**In Example 4.1, supposea a new state 15 is added to the gridworld just below state 13, and its actions, `left`,`up`,`right` and `down`, take the agent to states 12, 13, 14 and 15, respectively. Assume that the transitions _from_ the original states are unchanged. What, then, is _v<sub>π</sub>_ (15) for the equiprobable random policy? Noew suppose the dynamics of state 13 are also changed, such that action `down` from state 13 takes the agent to the new state 15. What is _v<sub>π</sub>_ (15) for the equiprobable random policy in this case?**
 
- _v<sub>π</sub>_ (15) = -1/4 * (22 + 20 + 14 + E [ G | 15 ]) = __18.67__
- _v<sub>π</sub>_ (13) = 


### Exercise 4.3
**What are the equations analogous to (4.3), (4.4), and (4.5) for the action value function q<sub>π</sub> and its successive approximation by a sequence of functions q<sub>0</sub>, q<sub>1</sub>, q<sub>2</sub>, ... ?**
 
