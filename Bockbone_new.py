import torch
import torch.nn as nn
import cv2

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def conv_3x3_bn(inp, oup, image_size, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )

# class BottleNeck(nn.Module):
#     expansion = 4
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#         self.residual_function = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             CBAM(out_channels),
#             nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels * BottleNeck.expansion),
#             CBAM(out_channels * BottleNeck.expansion),
#         )
#
#         self.shortcut = nn.Sequential()
#
#         if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(out_channels * BottleNeck.expansion),
#             )
#
#     def forward(self, x):
#         return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
        # return y

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=8):
        super(ChannelAttentionModule, self).__init__()
        #使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class Spa(nn.Module):
    def __init__(self, channel, rate=4):
        super(Spa, self).__init__()
        # context Modeling
        self.conv_mask = nn.Conv2d(channel, 1, kernel_size=1)

        self.conv2d = nn.Sequential(
            nn.Conv2d(channel, int(channel / rate), kernel_size=1),
            nn.BatchNorm2d(int(channel / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(channel / rate), int(channel / rate), kernel_size=7, stride=1, padding=3),
            nn.Conv2d(int(channel / rate), channel, kernel_size=1)
        )

        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        # self.batchnorm = nn.BatchNorm2d(2)

    def forward(self, x):
        #map尺寸不变，缩减通道
        # context Modeling
        batch, channel, height, width = x.shape
        # input_x = x
        # # [N, C, H * W]
        # input_x = input_x.view(batch, channel, height * width)
        # # # [N, 1, C, H * W]
        # input_x = input_x.unsqueeze(1)
        # # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
         # [N, 1, H * W]
        context_mask = self.softmax(context_mask)  # softmax操作
        # # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        context_mask = context_mask.view(batch, 1, height, width)
        # # [N, 1, C, 1]
        # context = torch.matmul(input_x, context_mask)
        # # [N, C, 1, 1]
        # context = context.view(batch, channel, 1, 1)
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = torch.matmul(out, context_mask)


        # context = context+x
        out = self.sigmoid(self.conv(out))
        # out = torch.matmul(input_x, out)
        #yuandaima
        # out = self.sigmoid(self.conv(self.conv2d(x)))
        # context = torch.matmul(out, context_mask)
        # context_mask = self.relu(context)

        return out * x

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x

# class CBAM(nn.Module):
#     def __init__(self, channel):
#         super(CBAM, self).__init__()
#         self.channel_attention = ChannelAttentionModule(channel)
#         # self.spatial_attention = SpatialAttentionModule()
#         self.spatial_attention = Spa(channel)
#         # self.conv = nn.Conv2d(channel, 1, kernel_size=1)
#     def forward(self, x):
#         out1 = self.channel_attention(x) * x
#         # out = self.spatial_attention(out1) * out1
#         out2 = self.spatial_attention(x) * x
#         out = out1 + out2
#         #
#         # out = out + x
#         return out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim,  dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return x

class MBConv(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),

                Spa(hidden_dim, hidden_dim),
                # SpatialAttentionModule(),

                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)

class Bockbone(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, embed_dims=[92, 192, 384],
                 decoder_embed_dim=512, block_types=['C', 'C', 'T', 'T']):
        super().__init__()
        ih, iw = image_size

        block = {'C': MBConv}

        self.s0 = self._make_layer(  #num_blocks = [2, 3, 2, 3, 5, 2]   # channels = [16, 64, 96, 192, 384, 768]
            conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih // 2, iw // 2))
        # self.s_0 = BottleNeck(channels[0], channels[1]).to(device)
        self.s1 = self._make_layer(  #(16, 64, 256, 192, 384, 512, 1024)
            block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4)).to(device)
        self.s2 = self._make_layer(
            block[block_types[0]], channels[1], channels[2], num_blocks[2], (ih // 8, iw // 8)).to(device)
        self.s3 = self._make_layer(
            block[block_types[0]], channels[2], channels[3], num_blocks[3], (ih // 16, iw // 16)).to(device)


    def forward(self, x):
        x = self.s0(x).to(device)
        # x_0 = self.s_0(x)

        x_1 = self.s1(x)

        x_2 = self.s2(x_1)

        x_3 = self.s3(x_2)

        return x_1, x_2, x_3

    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)


def bockbone_0():
    embed_dims = [256, 512, 1024]
    num_blocks = [2, 2, 2, 2]  # L
    channels = [64, 96, 192, 384, 768]  # D[16, 64, 256, 192, 384, 768]
    return Bockbone((224, 224), 3, num_blocks, channels, embed_dims=embed_dims, decoder_embed_dim=512)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    img = torch.randn(1, 3, 224, 224)

    net = bockbone_0()
    out = net(img)
    # print(out.shape, count_parameters(net))
    print( count_parameters(net))