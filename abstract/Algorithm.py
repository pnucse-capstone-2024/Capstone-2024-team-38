from abc import abstractmethod


class Algorithm():
    @abstractmethod
    def learn(self, total_episodes, rollout_callback=None, eval_callback=None):
        pass
    
    @abstractmethod
    def predict(self, state, deterministic=True):
        pass