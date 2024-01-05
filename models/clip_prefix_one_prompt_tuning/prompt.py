import torch
import torch.nn as nn
import copy

import math
def tensor_prompt(a, b, c=None,ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a, b), requires_grad=True)
        if ortho:
            nn.init.orthogonal_(p)
        else:
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a, b, c), requires_grad=True)
        if ortho:
            nn.init.orthogonal_(p)
        else:
            for index in range(a):
                nn.init.kaiming_uniform_(p[index], a=math.sqrt(5))
    return p



class PrefixKeqV(nn.Module):
    def __init__(self, emb_d, n_tasks, e_p_length, e_pool_size, e_layers):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d 
        
        self.n_tasks = n_tasks 
        self._init_smart( e_p_length,e_pool_size,e_layers)

        
        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            setattr(self, f'e_p_{e}', p)

    def _init_smart(self, e_p_length, e_pool_size, e_layers):
        self.e_layers = e_layers
        self.e_p_length = e_p_length
        self.e_pool_size = e_pool_size

    def process_task_count(self):
        self.task_count += 1

    def forward(self, l,batch_size, task_id=None):

        p_return = None

        if l in self.e_layers:
            
            p = getattr(self, f'e_p_{l}')  
            if task_id is None:
                return None
            if isinstance(task_id, int):
                P_ = p[task_id].expand(batch_size, -1, -1)
            else:
                P_ = p[task_id]

            p_return = P_
    
        return p_return

class PrefixKneqV(nn.Module):
    def __init__(self, emb_d, n_tasks, e_p_length, e_pool_size, e_layers):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d 
        
        self.n_tasks = n_tasks 
        self._init_smart( e_p_length,e_pool_size,e_layers)

        for e in self.e_layers:

            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            setattr(self, f'e_p_{e}', p)

    def _init_smart(self, e_p_length, e_pool_size, e_layers):

        self.e_layers = e_layers
        self.e_p_length = e_p_length
        self.e_pool_size = e_pool_size

    def process_task_count(self):
        self.task_count += 1

    def forward(self, l,batch_size, task_id=None):

        p_return = None
       
        if l in self.e_layers:
            
            p = getattr(self, f'e_p_{l}')  
            if task_id is None:
                return None
            if isinstance(task_id, int):
                P_ = p[task_id].expand(batch_size, -1, -1) 
            else:

                P_ = p[task_id]
            i = int(self.e_p_length/2)
            Ek = P_[:, :i, :].reshape((batch_size, -1, self.emb_d)) 
            Ev = P_[:, i:, :].reshape((batch_size, -1, self.emb_d)) 
            p_return = [Ek, Ev]
    
        return p_return


