import torch
import torch.nn as nn

class BYOLWrapper(nn.Module):
    def __init__(self, model, projector_dim):
        super().__init__()
        self.model = model
        self.predictor = nn.Linear(projector_dim, projector_dim, bias=False)
        nn.init.eye_(self.predictor.weight)

    def forward(self, x, use_predictor=False):
        if not use_predictor:
            return self.model(x)
        else:
            features = self.model.forward_features(x)
            try: # for resnet50-gn
                features = self.model.global_pool(features)
                x = self.predictor(features)
                x = self.model.fc(x)
            except: # for vitbase-ln
                features = features[:, 0]
                x = self.model.fc_norm(features)
                x = self.predictor(x)
                x = self.model.head(x)
            return x