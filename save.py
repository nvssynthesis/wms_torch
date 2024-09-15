from json import JSONEncoder
from torch.utils.data import Dataset
import torch
import json

class EncodeTensor(JSONEncoder,Dataset):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super(json.NpEncoder, self).default(obj)


def save_rt_model(model, fn: str):
    with open(fn, 'w') as json_file:
        json.dump(model.state_dict(), json_file, cls=EncodeTensor)