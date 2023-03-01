import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def build_model(self, cfg):
        raise NotImplementedError

    def save_model_weights(self, states):
        #expects a new state with "models" key
        states["model"] = self.state_dict() 
        return states

    def load_weights(self, states):
        self.load_state_dict(states)
