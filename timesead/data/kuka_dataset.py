#add root to path
import sys
sys.path.append('/home/alessio/alessio/Research/GAN/Robot/Repositories/TimeSeAD')

from typing import Tuple, Optional, Dict, Any, Union, List, Callable
from timesead.data.dataset import BaseTSDataset
from timesead.utils.metadata import DATA_DIRECTORY
from RoADDataset import Dataset
import torch

class KukaDataset(BaseTSDataset):
    def __init__(self,training:bool,otherSet=None):
        self.training = training
        self.otherSet = otherSet
        dataset = Dataset()
        if training:
            subset = ['training']
        elif training == False and otherSet =='collision':
            subset = ['collision']
        else:
            subset = ['control',otherSet]
        self.dataset = []
        for x in subset:
            if x != 'control':
                self.dataset.extend(dataset.sets[x])
            else:
                for sset in dataset.sets['collision']:
                    collision = torch.tensor(sset[:,-1],dtype=torch.float32)
                    collision = torch.roll(collision, -2, 0)>0
                    tmp=None
                    for t in range(0,collision.shape[0]):
                        sset[t,-1] = collision[t]
                        if collision[t] == 1:
                            if tmp is not None:
                                self.dataset.append(tmp)
                            tmp=None
                        elif tmp is None:
                            tmp = torch.tensor(sset[t:t+1])
                        else:
                            toAdd=torch.tensor(sset[t:t+1])
                            tmp = torch.cat((tmp,toAdd),dim=0)

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def seq_len(self) -> Union[int, List[int]]:
        return [x.shape[0] for x in self.dataset]

    @property
    def num_features(self) -> Union[int, Tuple[int, ...]]:
        return 86

    @staticmethod
    def get_default_pipeline() -> Dict[str, Dict[str, Any]]:

        return {
        }

    @staticmethod
    def get_feature_names() -> List[str]:
        return ['']*86

    def __getitem__(self, index: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        time_series = self.dataset[index]
        s=time_series.shape
        if self.training:
            time_series = time_series
            target = torch.zeros(s[0],1)
        else:
            target = time_series[:,-1]
            time_series = time_series[:,:-1]

        time_series_tensor = torch.tensor(time_series, dtype=torch.float)
        target_tensor = torch.tensor(target, dtype=torch.float)
        target_tensor = torch.roll(target_tensor, -2, 0)>0
        target_tensor = target_tensor.float()
        return (time_series_tensor,), (target_tensor,)