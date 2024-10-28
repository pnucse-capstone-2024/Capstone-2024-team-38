from collections import namedtuple
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from abstract.Model import Model

PPOBuffer = namedtuple('PPOBuffer', ['action', 'state', 'logprobs', 'reward', 'state_values', 'is_terminals'])

class RolloutBuffer: 
    def __init__(self):
        self.buffer = []
 
    def append(self, action, state, logprobs, reward, state_values, is_terminals): 
        self.buffer.append(PPOBuffer(action, state, logprobs, reward, state_values, is_terminals))
    
    def clear(self):
        del self.buffer[:]

    def split_batches(self, batch_size):
        for i in range(0, len(self.buffer), batch_size):
            yield PPOBuffer(*zip(*self.buffer[i:i+batch_size]))

    def __getattr__(self, name: str) -> torch.Any:
        idx = PPOBuffer._fields.index(name)
        if name in ['action', 'state', 'logprobs', 'state_values']:
            return [b[idx].tolist() for b in self.buffer]
        else:
            return [b[idx] for b in self.buffer]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init, device = "cpu", init_weight=None):
        super(ActorCritic, self).__init__()

        self.device = device

        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init)

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        if init_weight is not None:
            self.actor.apply(init_weight)
            self.critic.apply(init_weight)

    def forward(self):
        raise NotImplementedError
    
    def act(self, state, deterministic=False):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = torch.clamp(dist.sample(), -1, 1).to(self.device)
        action_logprobs = dist.log_prob(action)
        state_val = self.critic(state)

        if deterministic:
            return action_mean.detach(), action_logprobs.detach(), state_val.detach()

        return action.detach(), action_logprobs.detach(), state_val.detach()
    
    def evaluate(self, state, action):

        state = state.to(self.device)
        action = action.to(self.device)

        action_mean = self.actor.to(self.device)(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = action.reshape(-1, self.action_dim)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_val = self.critic.to(self.device)(state)

        return action_logprobs, state_val, dist_entropy
    
class PPO(Model):
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 actor_lr, 
                 critic_lr, 
                 gamma, 
                 epochs, 
                 eps_clip, 
                 seed=22,
                 batchSize=128,
                 action_std_init=0.6, 
                 device="cpu",
                 init_weight=None):
        self.action_std = action_std_init
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs
        self.device = device
        self.seed = seed
        self.batchSize = batchSize

        torch.manual_seed(seed)

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, action_std_init, device, init_weight)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': actor_lr},
            {'params': self.policy.critic.parameters(), 'lr': critic_lr}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init, device, init_weight)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            action, action_logprobs, state_val = self.policy_old.act(state, deterministic)
            action = action.detach().cpu().numpy().flatten()

            return action, action_logprobs, state_val
    
    def action_for_step(self, action):
        return action
    
    def push_memory(self, *args, **kwargs):
        try:
            self.buffer.append(action=kwargs["action"], 
                            state=kwargs["state"], 
                            logprobs=kwargs["logprobs"], 
                            reward=kwargs["reward"], 
                            state_values=kwargs["state_values"], 
                            is_terminals=kwargs["is_terminals"])
        except:
            raise ValueError("Arguments must be a dictionary with keys: action, state, logprobs, reward, state_values, is_terminals")
        
    def optimize_model(self):
        for _ in range(self.epochs):
            for batch in self.buffer.split_batches(self.batchSize):
                rewards = []
                discounted_reward = 0
                for reward, is_terminal in zip(reversed(batch.reward), reversed(batch.is_terminals)):
                    if is_terminal:
                        discounted_reward = 0
                    discounted_reward = reward + (self.gamma * discounted_reward)
                    rewards.insert(0, discounted_reward)

                rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

                old_states = torch.squeeze(torch.tensor(batch.state, dtype=torch.float32)).detach().to(self.device)
                old_actions = torch.squeeze(torch.tensor(batch.action, dtype=torch.float32)).detach().to(self.device)
                old_logprobs = torch.squeeze(torch.tensor(batch.logprobs, dtype=torch.float32)).detach().to(self.device)
                old_state_values = torch.squeeze(torch.tensor(batch.state_values, dtype=torch.float32)).detach().to(self.device)

                advantages = rewards.detach() - old_state_values.detach()

                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

                state_values = torch.squeeze(state_values)

                ratios = torch.exp(logprobs - old_logprobs.detach())

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.buffer.clear()