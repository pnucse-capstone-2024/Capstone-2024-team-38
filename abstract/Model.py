from abc import abstractmethod


class Model():
    @abstractmethod
    def select_action(self, state):
        raise NotImplementedError("select_action method must be implemented")
    
    @abstractmethod
    def action_for_step(self, action):
        return action
    
    @abstractmethod
    def optimize_model(self):
        raise NotImplementedError("optimize_model method must be implemented")
    
    @abstractmethod
    def push_memory(self, *args, **kwargs):
        pass