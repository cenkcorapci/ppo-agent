# Proximal Policy Optimization
- TRPO mindset; To improve training stability, we should avoid parameter updates that change the policy too much at one step. Trust region policy optimization (TRPO) (Schulman, et al., 2015) carries out this idea by enforcing a KL divergence constraint on the size of policy update at each iteration.
- Policy Gradient methods have convergence problem which is addressed by the natural policy gradient.
- The central idea of Proximal Policy Optimization is to avoid having too large policy update.
- PPO uses a slightly different approach. Instead of imposing a hard constraint, it formalizes the constraint as a penalty in the objective function. 
- Minorize-Maximization MM algorithm
- Line Search; find the steepest direction, move towards a step 
- Trust region; finda local area to be safe so we don't step too far ahead and fall, then find
the optimal point in the area and continue from there.
- Trust region can be set dynamically by considering the amount og change in policy gradient.
- We limit how far we can change our policy by KL divergence.
- We use the advantage function instead of the expected reward because it reduces the variance of the estimation.
-  The objective function of PPO takes the minimum one between the original value and the clipped version and therefore we lose the motivation for increasing the policy update to extremes for better rewards.
- With the idea of importance sampling, we can evaluate a new policy with samples collected from an older policy. This improves sample efficiency.
- Effectively, this discourages large policy change if it is outside our comfortable zone.
```
Q-learning (with function approximation) fails on many simple problems and is poorly understood, 
vanilla policy gradient methods have poor data efficiency and robustness; and trust region policy optimization (TRPO) is relatively complicated, and is not compatible with architectures that include noise (such as dropout) or parameter sharing (between the policy and value function, or with auxiliary tasks).
```
- PPO adds a soft constraint that can be optimized by a first-order optimizer. We may make some bad decisions once a while but it strikes a good balance on the speed of the optimization. Experimental results prove that this kind of balance achieves the best performance with the most simplicity.