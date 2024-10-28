from abstract.Algorithm import Algorithm
from models.PPO_MAXA import PPO
from tqdm import tqdm

class Origin_PPO_MAXA(Algorithm):
    def __init__(self,
                 env, 
                 actor_lr, 
                 critic_lr, 
                 gamma, 
                 epochs, 
                 eps_clip, 
                 seed=22,
                 action_std_init=0.6, 
                 rollout_length=2048,
                 init_weight=None,
                 device="cpu"):
        self.env = env
        self.rollout_length = rollout_length
        self.seed = seed
        self.Model = PPO(
                        state_dim=env.observation_space.shape[0],
                        action_dim=env.action_space.shape[0],
                        actor_lr=actor_lr,
                        critic_lr=critic_lr,
                        gamma=gamma,
                        epochs=epochs,
                        eps_clip=eps_clip,
                        action_std_init=action_std_init,
                        seed=seed,
                        init_weight=init_weight,
                        device=device)

    def learn(self, total_timestamp, rollout_callback=None, eval_callback=None):
        real_rollout_count = total_timestamp // self.rollout_length + (total_timestamp % self.rollout_length > 0)
        real_total_timestamp = real_rollout_count * self.rollout_length
        state = None
        done = True
        temp_timestamp = 0
        rollout_reward = 0
        rollout_rewards = []
        timestamp = 0
        for _ in tqdm(range(real_total_timestamp)):
            if done:
                state, _ = self.env.reset(seed=self.seed)
                done = False

            action, action_logprobs, state_val = self.Model.select_action(state)
            action_for_step = self.Model.action_for_step(action)
            next_state, reward, done, _, _ = self.env.step(action_for_step)
            
            self.Model.push_memory(action=action,
                                   state=state,
                                   logprobs=action_logprobs,
                                   reward=reward,
                                   state_values=state_val,
                                   is_terminals=done)

            state = next_state
            rollout_reward += reward
            temp_timestamp += 1
            timestamp += 1

            if eval_callback is not None:
                eval_callback(reward, timestamp)

            if temp_timestamp == self.rollout_length:
                self.Model.optimize_model()
                if rollout_callback is not None:
                    rollout_callback(rollout_reward)
                rollout_rewards.append(rollout_reward)
                temp_timestamp = 0
                rollout_reward = 0

        return rollout_rewards
    
    def predict(self, state, deterministic=True):
        action, _, _ = self.Model.select_action(state, deterministic)
        return self.Model.action_for_step(action), None