# ====================   Define custom activations ==============================
import torch
import math

def dosomething(self):
    print("hello world")

class Bell_reLU(torch.nn.Module): 
    def __init__(self): 
        super(Bell_reLU, self).__init__() 
        self.act_pos_lambda = lambda a : (1 / ( 1 + math.e**(-(-a + math.e) * math.e) ) )
        self.act_neg_lambda = lambda a : (1 / ( 1 + math.e**(-(a + math.e) * math.e) ) )
  
    def forward(self, a, beta=1): 
        output = torch.where(a < 0, self.act_neg_lambda(a), self.act_pos_lambda(a))
        return output