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


In the Lunar Lander environment, the agent's task is to learn how to land a lunar module safely on the moon's surface. This requires the agent to balance fuel efficiency and safety considerations. The agent needs to learn from its past experiences, developing a strategy to approach the landing pad while minimizing its speed and using as little fuel as possible.

All reinforcement learning (RL) methods will be built from scratch, providing a comprehensive understanding of their workings and we will use PyTorch to build our neural network model.

Let's initialize a LunarLander-v2 environmnet, make random actions in the environment, then view a recording of it.









---------------------------------------------------------------------------------------------------------------------------------------------
## Deep Q-Learning and Variants in Gym's Lunar Lander Environment

In the Lunar Lander environment, the agent's task is to learn how to land a lunar module safely on the moon's surface. This requires the agent to balance fuel efficiency and safety considerations. The agent needs to learn from its past experiences, developing a strategy to approach the landing pad while minimizing its speed and using as little fuel as possible.

All reinforcement learning (RL) methods will be built from scratch, providing a comprehensive understanding of their workings and we will use PyTorch to build our neural network model.

Let's initialize a LunarLander-v2 environmnet, make random actions in the environment, then view a recording of it.

Function Signature:

python
Copy code
def play_episode(env, agent, seed=42):
env: This parameter represents the Gym environment in which the episode will be played.
agent: This parameter represents the agent that will interact with the environment.
seed=42: This parameter sets the random seed for reproducibility. Default is set to 42.
Docstring:

python
Copy code
'''
Plays a full episode for a given agent, environment and seed.
'''
This provides a brief description of what the function does.
Variable Initialization:

python
Copy code
score = 0
observation = env.reset(seed=seed)
score: This variable keeps track of the total reward obtained during the episode. It starts from 0.
observation: This variable holds the initial observation obtained by resetting the environment. The reset() method initializes the environment to its starting state and returns the initial observation.
Episode Execution:

python
Copy code
while True:
    action = agent.act(observation)
    observation, reward, done, info = env.step(action)

    score += reward

    # End the episode if done
    if done:
        break
This while loop runs until the episode is terminated (done becomes True).
Inside the loop:
The agent selects an action (action) based on the current observation (observation) using its act() method.
The environment takes a step (env.step(action)) using the selected action, which returns the next observation (observation), the reward obtained (reward), whether the episode is done (done), and additional information (info).
The reward obtained is added to the total score (score += reward).
If the episode is done (done == True), the loop breaks, and the episode ends.
Return:

python
Copy code
return score
After the episode ends, the function returns the total score obtained during the episode.
This function essentially simulates an episode of interaction between the provided agent and environment, returning the total reward obtained during the episode. It's useful for evaluating the performance of an agent in a specific environment.


https://i.imgur.com/qFNn9ai.gif

![](https://i.imgur.com/qFNn9ai.gif)

#### Observations:
- The safe agent may not have hit the ground, but it didn't take long to fly off screen, due to its inability to use the side engines.

---

## The Stable Agent
Let's try to define and agent that can remain stable in the air.

It will operate via the following rules:

1. If below height of 1: action = 2 (main engine)
2. If angle is above π/50: action = 1 (fire right engine)
3. If angle is above π/50: action = 1 (fire left engine)
4. If x distance is above 0.4: action = 3 (fire left engine)
5. If x distance is below -0.4: action = 1 (fire left engine)
6. If below height of 1.5: action = 2 (main engine)
6. Else: action = 0 (do nothing)

The idea is the lander will always use its main engine if it falls below a certain height, next it will prioritize stabilizing the angle of the lander, then the distance, then keeping it above another height.

![](https://i.imgur.com/Bdq1Hdl.gif)

#### Observations:
- Crafting a straightforward set of rules to guide the lunar lander is more challenging than anticipated.
- Our initial efforts achieved some stability, but eventually, the lander lost control.

---

# Deep Reinforcement Learning
To address this challenge, we'll use deep reinforcement learning techniques to train an agent to land the spacecraft.

Simpler tabular methods are limited to discrete observation spaces, meaning there are a finite number of possible states. In `LunarLander-v2` however, we're dealing with a continuous range of states across 8 different parameters, meaning there are a near-infinite number of possible states. We could try to bin similar values into groups, but due to the sensitive controls of the game, even slight errors can lead to significant missteps.

To get around this, we'll use a `neural network Q-function approximator`. This lets us predict the best actions to take for a given state, even when dealing with a vast number of potential states. It's a much better match for our complex landing challenge.


The algorithm:

1. **Initialization**: Begin by initializing the parameters for two neural networks, $Q(s,a)$ (referred to as the online network) and $\hat{Q}(s,a)$ (known as the target network), with random weights. Both networks serve the function of mapping a state-action pair to a Q-value, which is an estimate of the expected return from that pair. Also, set the exploration probability $\epsilon$ to 1.0, and create an empty replay buffer to store past transition experiences.
2. **Action Selection**: Utilize an epsilon-greedy strategy for action selection. With a probability of $\epsilon$, select a random action $a$, but in all other instances, choose the action $a$ that maximizes the Q-value, i.e., $a = argmax_aQ(s,a)$.
3. **Experience Collection**: Execute the chosen action $a$ within the environment emulator and observe the resulting immediate reward $r$ and the next state $s'$.
4. **Experience Storage**: Store the transition $(s,a,r,s')$ in the replay buffer for future reference.
5. **Sampling**: Randomly sample a mini-batch of transitions from the replay buffer for training the online network.
6. **Target Computation**: For every transition in the sampled mini-batch, compute the target value $y$. If the episode has ended at this step, $y$ is simply the reward $r$. Otherwise, $y$ is the sum of the reward and the discounted estimated optimal future Q-value, i.e.,  $y = r + \gamma \max_{a' \in A} \hat{Q}(s', a')$
7. **Loss Calculation**: Compute the loss, which is the squared difference between the Q-value predicted by the online network and the computed target, i.e., $\mathcal{L} = (Q(s,a) - y)^2$
8. **Online Network Update**: Update the parameters of the online network $Q(s,a)$ using Stochastic Gradient Descent (SGD) to minimize the loss.
9. **Target Network Update**: Every $N$ steps, update the target network by copying the weights from the online network to the target network $\hat{Q}(s,a)$.
10. **Iterate**: Repeat the process from step 2 until convergence.

### Defining the Deep Q-Network
Our network will be a simple feedforward neural network that takes the state as input and produces Q-values for each action as output. For `LunarLander-v2` the state is an 8-dimensional vector and there are 4 possible actions.


### Defining the Replay Buffer
In the context of RL, we employ a structure known as the replay buffer, which utilizes a deque. The replay buffer stores and samples experiences, which helps us overcome the problem of *step correlation*.

A *deque* (double-ended queue) is a data structure that enables the addition or removal of elements from both its ends, hence the name. It is particularly useful when there is a need for fast append and pop operations from either end of the container, which it provides at O(1) time complexity. In contrast, a list offers these operations at O(n) time complexity, making the deque a preferred choice in cases that necessitate more efficient operations.

Moreover, a deque allows setting a maximum size. Once this maximum size is exceeded during an insertion (push) operation at the front, the deque automatically ejects the item at the rear, thereby maintaining its maximum length.

In the replay buffer, the `push` method is utilized to add an experience. If adding this experience exceeds the maximum buffer size, the oldest (rear-most) experience is automatically removed. This approach ensures that the replay buffer always contains the most recent experiences up to its capacity.

The `sample` method, on the other hand, is used to retrieve a random batch of experiences from the replay buffer. This randomness is critical in breaking correlations within the sequence of experiences, which leads to more robust learning.

This combination of recency and randomness allows us to learn on new training data, without training samples being highly correlated.


### Define the DQN Agent
The DQN agent handles the interaction with the environment, selecting actions, collecting experiences, storing them in the replay buffer, and using these experiences to train the network. Let's walk through each part of this process:

#### Initialisation
The `__init__` function sets up the agent:

- `self.device`: We start by checking whether a GPU is available, and, if so, we use it, otherwise, we fall back to CPU.
- `self.gamma`: This is the discount factor for future rewards, used in the Q-value update equation.
- `self.batch_size`: This is the number of experiences we'll sample from the memory when updating the model.
- `self.q_network` and `self.target_network`: These are two instances of the Q-Network. The first is the network we're actively training, and the second is a copy that gets updated less frequently. This helps to stabilize learning.
- `self.optimizer`: This is the optimization algorithm used to update the Q-Network's parameters.
- `self.memory`: This is a replay buffer that stores experiences. It's an instance of the `ReplayBuffer` class.

#### Step Function
The `step` function is called after each timestep in the environment:

- The function starts by storing the new experience in the replay buffer.
- If enough experiences have been stored, it calls `self.update_model()`, which triggers a learning update.

#### Action Selection
The act function is how the agent selects an action:

- If a randomly drawn number is greater than $\epsilon$, it selects the action with the highest predicted Q-value. This is known as exploitation: the agent uses what it has learned to select the best action.
- If the random number is less than $\epsilon$, it selects an action randomly. This is known as exploration: the agent explores the environment to learn more about it.

#### Model Update
The `update_model` function is where the learning happens:

- It starts by sampling a batch of experiences from the replay buffer.
- It then calculates the current Q-values for the sampled states and actions, and the expected - Q-values based on the rewards and next states.
- It calculates the loss, which is the mean squared difference between the current and expected Q-values.
- It then backpropagates this loss through the Q-Network and updates the weights using the optimizer.

#### Target Network Update
Finally, the `update_target_network` function copies the weights from the Q-Network to the Target Network. This is done periodically (not every step), to stabilize the learning process. Without this, the Q-Network would be trying to follow a moving target, since it's learning from estimates produced by itself.

### Training the Agent

Training the agent involves having the agent interact with the `LunarLander-v2` environment over a sequence of steps. Over each step, the agent receives a state from the environment, selects an action, receives a reward and the next state, and then updates its understanding of the environment (the Q-table in the case of Q-Learning).

The `train` function orchestrates this process over a defined number of episodes, using the methods defined in the DQNAgent class. Here's how it works:

#### Initial Setup
- `scores`: This list stores the total reward obtained in each episode.
- `scores_window`: This is a double-ended queue with a maximum length of 100. It holds the scores of the most recent 100 episodes and is used to monitor the agent's performance.
-`eps`: This is the epsilon for epsilon-greedy action selection. It starts from `eps_start` and decays after each episode until it reaches `eps_end`.

#### Episode Loop
The training process runs over a fixed number of episodes. In each episode:

- The environment is reset to its initial state.
- he agent then interacts with the environment until the episode is done (when a terminal state is reached).

#### Step Loop
In each step of an episode:

- The agent selects an action using the current policy (the act method in `DQNAgent`).
The selected action is applied to the environment using the step method, which returns the next state, the reward, and a boolean indicating whether the episode is done.
- The agent's step method is called to update the agent's knowledge. This involves adding the experience to the replay buffer and, if enough experiences have been collected, triggering a learning update.
- The state is updated to the next state, and the reward is added to the score.

After each episode:

- The score for the episode is added to `scores` and `scores_window`.
- Epsilon is decayed according to `eps_decay`.
- If the episode is a multiple of `target_update`, the target network is updated with the latest weights from the Q-Network.
- Finally, every 100 episodes, the average score over the last 100 episodes is printed.

The function returns the list of scores for all episodes.

This training process, which combines experiences from the replay buffer and separate target and Q networks, helps to stabilize the learning and leads to a more robust policy.


![](https://i.imgur.com/NAg48Qk.gif)

## Double DQN (DDQN)
The Double Deep Q-Network (DDQN) algorithm is a modification of the standard Deep Q-Network (DQN) algorithm, which reduces the overestimation bias in the Q-values, thereby improving the stability of the learning process. You can read the original publication by Hasselt et al from late 2015 here:

https://arxiv.org/abs/1509.06461

### The DDQN Algorithm

1. **Initialization**: Similar to DQN, initialize the parameters of two neural networks, $Q(s,a)$ (online network) and $\hat{Q}(s,a)$ (target network), with random weights. Both networks estimate Q-values from state-action pairs. Also, set the exploration probability $\epsilon$ to 1.0, and create an empty replay buffer.

2. **Action Selection**: Use an epsilon-greedy strategy, just like in DQN. With a probability of $\epsilon$, select a random action $a$, otherwise, select the action $a$ that yields the highest Q-value, i.e., $a = argmax_aQ(s,a)$.

3. **Experience Collection**: Carry out the selected action $a$ in the environment to get the immediate reward $r$ and the next state $s'$.

4. **Experience Storage**: Store the transition tuple $(s,a,r,s')$ in the replay buffer.

5. **Sampling:** Randomly sample a mini-batch of transitions from the replay buffer.

6. **Target Computation**: Here comes the primary difference from DQN. For every transition in the sampled mini-batch, compute the target value $y$. If the episode has ended, $y = r$. Otherwise, unlike DQN that uses the max operator to select the action from the target network, DDQN uses the online network to select the best action, and uses its Q-value estimate from the target network, i.e., $y = r + \gamma \hat{Q}(s', argmax_{a' \in A} Q(s', a'))$. This double estimator approach helps to reduce overoptimistic value estimates.

7. **Loss Calculation**: Compute the loss as the squared difference between the predicted Q-value from the online network and the computed target, i.e., $\mathcal{L} = (Q(s,a) - y)^2$.

8. **Online Network Update**: Perform Stochastic Gradient Descent (SGD) on the online network to minimize the loss.

9. **Target Network Update**: Every $N$ steps, update the target network by copying the weights from the online network.

10. **Iterate**: Repeat the process from step 2 until convergence.

In summary, the key difference in DDQN lies in the way the target Q-value is calculated for non-terminal states during the update. DDQN chooses the action using the online network and estimates the Q-value for this action using the target network. This modification helps mitigate the issue of overestimation present in standard DQN.




![](https://i.imgur.com/rrfB9Vl.gif)

## Dueling Deep Q-Networks (Dueling DQN)



### The Dueling DQN Algorithm
1. **Initializatin**: In Dueling DQN, initialize the parameters of two neural networks, $Q(s,a)$ (online network) and $\hat{Q}(s,a)$ (target network), with random weights. Unlike the traditional DQN, each network in Dueling DQN splits into two separate streams at some point - one for estimating the state-value function $V(s)$ and the other for estimating the advantage function $A(s,a)$. Also, set the exploration probability $\epsilon$ to 1.0, and create an empty replay buffer.

2. **Action Selection**: The action selection process is the same as DQN. Use an epsilon-greedy strategy. With a probability of $\epsilon$, select a random action $a$, otherwise, select the action $a$ that yields the highest Q-value, i.e., $a = argmax_aQ(s,a)$.

3. **Experience Collection**: Carry out the selected action $a$ in the environment to obtain the immediate reward $r$ and the next state $s'$.

4. **Experience Storage**: Store the transition tuple $(s,a,r,s')$ in the replay buffer.

5. **Sampling**: Randomly sample a mini-batch of transitions from the replay buffer.

6. **Target Computation**: For each transition in the sampled mini-batch, compute the target value $y$. If the episode has ended, $y = r$. Otherwise, compute $y$ as $y = r + \gamma \hat{Q}(s', argmax_{a' \in A} Q(s', a'))$.

7. **Loss Calculation**: Compute the loss as the squared difference between the predicted Q-value from the online network and the computed target, i.e., $\mathcal{L} = (Q(s,a) - y)^2$.

8. **Online Network Update**: Use Stochastic Gradient Descent (SGD) or another optimization algorithm to update the online network and minimize the loss.

9. **Target Network Update**: Every $N$ steps, update the target network by copying the weights from the online network.

10. **Iterate**: Repeat the process from step 2 until convergence.

Dueling DQN indeed introduces a novel network architecture for approximating the Q-value function. It separates the Q-value into two parts: the state-value function $V(s)$, which estimates the value of a state regardless of the actions, and the advantage function $A(s,a)$, which measures the relative advantage of taking an action in a state compared to the other actions.

At first glance, it might seem logical to compute the Q-value simply by adding the state-value and the advantage: $Q(s,a) = V(s) + A(s,a)$. However, this equation presents an issue: it's underdetermined. There are infinite possible combinations of $V(s)$ and $A(s,a)$ that satisfy this equation for a given $Q(s,a)$. For instance, if the actual value of $Q(s,a)$ is 10, we would have the equation $10 = V(s) + A(s,a)$, for which there are infinite solutions.

The authors of the Dueling DQN paper propose a clever way to overcome this issue: they force the advantage function to have zero advantage at the chosen action. This means that the highest advantage, $A(s,a)$, is 0, and other advantages are negative or zero, thus providing a unique solution. To implement this, they modify the equation as follows:

$$ Q(s,a) = V(s)+(A(s,a) − \max_{a'}A(s, a') $$

This equation means that the Q-value is computed as the state-value $V(s)$ plus the difference between the advantage of the action $a$ and the maximum advantage over all possible actions in state $s$. In other words, the Q-value is now the value of the state plus the relative advantage of taking the action $a$ over the other actions. This mechanism provides a clear way to train the network and allows Dueling DQN to learn efficiently about state values and action advantages.

To implement this, we can use the original DQN algorithm and our original DQNAgent class, we just need to change the DQN it uses, in total just 2 lines of code changes in the agent class.



