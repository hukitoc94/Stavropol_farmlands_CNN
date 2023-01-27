
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torchvision
from torchvision import  models
from torch.nn import functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        batchNorm_momentum = 0.1
        self.stage1 = nn.Sequential( 
                nn.Conv2d(4, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64, momentum= batchNorm_momentum),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64, momentum= batchNorm_momentum),
                nn.ReLU()
            )
        self.stage2 = nn.Sequential( 
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum= batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum= batchNorm_momentum),
            nn.ReLU()
        )
        self.stage3 = nn.Sequential( 
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum= batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum= batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum= batchNorm_momentum),
            nn.ReLU()
        )
        self.stage4 = nn.Sequential( 
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum= batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum= batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum= batchNorm_momentum),
            nn.ReLU()
        )
        self.stage5 = nn.Sequential( 
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum= batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum= batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024, momentum= batchNorm_momentum),
            nn.ReLU()
        )
        self.stage5_de = nn.Sequential( 
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum= batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum= batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum= batchNorm_momentum),
            nn.ReLU()
        )
        self.stage4_de = nn.Sequential( 
            nn.Conv2d(2 * 512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum= batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum= batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum= batchNorm_momentum),
            nn.ReLU()
        )
        self.stage3_de = nn.Sequential( 
            nn.Conv2d(2 * 256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum= batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum= batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum= batchNorm_momentum),
            nn.ReLU()
        )
        self.stage2_de = nn.Sequential( 
            nn.Conv2d(2 * 128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum= batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum= batchNorm_momentum),
            nn.ReLU()
            
        )
        self.stage1_de = nn.Sequential( 
            nn.Conv2d(2 * 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum= batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
         #   nn.BatchNorm2d(label_nbr, momentum= batchNorm_momentum),
         #   nn.ReLU(),
           # nn.Conv2d(label_nbr, label_nbr, kernel_size=3, padding=1),
          #  nn.BatchNorm2d(label_nbr, momentum= batchNorm_momentum)
                  )


    def forward(self, x):
        # encoder
        e1 = self.stage1(x)
        e1_1, id1 = F.max_pool2d(e1,kernel_size=2, stride=2,return_indices=True)
        e2 = self.stage2(e1_1)
        e2_1, id2 = F.max_pool2d(e2,kernel_size=2, stride=2,return_indices=True)
        e3 = self.stage3(e2_1)
        e3_1, id3 = F.max_pool2d(e3,kernel_size=2, stride=2,return_indices=True)
        e4 = self.stage4(e3_1)
        e4_1, id4 = F.max_pool2d(e4,kernel_size=2, stride=2,return_indices=True)
        # bottleneck
        b = self.stage5(e4_1)
        b = self.stage5_de(b)

        # decoder
        d4 = F.max_unpool2d(b, id4, kernel_size=2, stride=2)
        d4 = self.stage4_de(torch.cat([e4,d4],1) )
        d3 = F.max_unpool2d(d4, id3, kernel_size=2, stride=2)
        d3 = self.stage3_de(torch.cat([e3,d3],1) )
        d2 = F.max_unpool2d(d3, id2, kernel_size=2, stride=2)
        d2 = self.stage2_de(torch.cat([e2,d2],1) )
        d1 = F.max_unpool2d(d2, id1, kernel_size=2, stride=2)
        d1 = self.stage1_de(torch.cat([e1,d1],1) )  
        return d1

