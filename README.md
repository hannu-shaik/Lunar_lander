OpenAI maintains gym, a Python library for experimenting with reinforcement learning techniques. Gym contains a variety of environments, each with their own characteristics and challenges. For this project, I used the LunarLander-v2 environment. The objective of LunarLander is to safely land a spaceship between two flags. The LunarLanader environment is:

Fully Observable: All necessary state information is known observed at every frame.
Single Agent: There is no competition or cooperation.
Deterministic: There is no stochasticity in the effects of actions or the rewards obtained.
Episodic: The reward is dependent only on the current state and action.
Discrete Action Space: There are only 4 discrete actions: thrust, left, right, nothing.
Static: There is no penalty or state change during action deliberation.
Finite Horizon: The episode terminates after a successful land, crash, or 1000 steps.
Mixed Observation Space: Each observation contains 8 values:
(Continuous): X distance from target site
(Continuous): Y distance from target site
(Continuous): X velocity
(Continuous): Y velocity
(Continuous): Angle of ship
(Continuous): Angular velocity of ship
(Binary): Left leg is grounded
(Binary): Right leg is grounded
Finally, after each step a reward is granted. The total reward of an episode is the sum of the rewards for all steps within that episode. The reward for moving from the top of the screen to landing pad with zero speed is awarded between 100 and 140 points. If the lander moves away from the landing pad it loses the same reward as moving the same distance towards the pad. The episode receives additional -100 or +100 points for crashing or landing, respectively. Grounding a leg is worth 10 points and thrusting the main engine receives -0.3 points. An episode score of 200 or more is considered a solution.

<img width="1414" alt="Screenshot 2024-03-06 at 10 27 07 PM" src="https://github.com/hannu-shaik/Lunar_lander/assets/140539636/c057ea96-1caa-4df3-abef-9047c7aa6c28">

II Experiment Design: DQN Algorithm & Model Architecture
<img width="909" alt="Screenshot 2024-03-06 at 10 28 34 PM" src="https://github.com/hannu-shaik/Lunar_lander/assets/140539636/e78e17cc-0d5d-462c-b922-7be3eebe7a74">
<img width="552" alt="Screenshot 2024-03-06 at 10 29 20 PM" src="https://github.com/hannu-shaik/Lunar_lander/assets/140539636/8adb7d26-aabe-4a72-b8cf-cc2f62f8eaa3">
<img width="533" alt="Screenshot 2024-03-06 at 10 30 15 PM" src="https://github.com/hannu-shaik/Lunar_lander/assets/140539636/471cc7a5-064c-4021-83fa-9739f1f6d194">
<img width="1134" alt="Screenshot 2024-03-06 at 10 30 54 PM" src="https://github.com/hannu-shaik/Lunar_lander/assets/140539636/e375f092-af07-4e79-8e12-516b84ced119">


