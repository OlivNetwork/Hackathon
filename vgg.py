import torch
from torch import nn


class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.c11 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.ReLU(True)
        )

        self.c12 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(True)
        )

        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.dropout = nn.Dropout(p=0.5)
        self.maxunpool2d = nn.MaxUnpool2d(2, stride=2)

        self.c21 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(True)
        )

        self.c22 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(True)
        )

        self.c31 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(True)
        )

        self.c32 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(True)
        )

        self.c41 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.c42 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.c43 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(True)
        )

        self.c51 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(True)
        )

        self.c52 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(True)
        )

    def forward(self, x):
        r11 = self.c11(x)  # 1, 64, 74, 100
        r12 = self.c12(r11)
        p1, indices1 = self.maxpool2d(r12)  # 1, 64, 37, 50   style1
        r21 = self.c21(p1)  # 1, 128, 37, 50
        r22 = self.c22(r21)
        p2, indices2 = self.maxpool2d(r22)  # 1, 128, 18, 25  style2
        r31 = self.c31(p2)  # 1, 256, 18, 25
        r32 = self.c32(r31)
        p3, indices3 = self.maxpool2d(r32)  # 1, 256, 9, 12   style3
        r41 = self.c41(p3)  # 1, 512, 9, 12
        r42 = self.c42(r41)
        r43 = self.c43(r42)
        p4, indices4 = self.maxpool2d(r43)  # 1, 512, 4, 6    style4  content1  sc41
        r51 = self.c51(p4)  # 1, 512, 4, 6
        r52 = self.c52(r51)
        p5, indices5 = self.maxpool2d(r52)  # 1, 512, 2, 3    style5  content2  sc52
        # x61 = self.c61(outputc5)  # 1, 2048, 1, 2
        # outputc6 = self.dropout(x61)  # 1, 2048, 1, 2              content3

        return r11, r21, r31, r41, r51, r42

