import torch.nn.functional as F
from My_model.Bockbone import bockbone_0
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.helpers import load_pretrained
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.registry import register_model
from My_model.utils import *
from My_model.Vgg_duibi import vgg_0

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  #16,196,92
        x = self.norm(x)  #16,196,92
        # H, W = H // self.patch_size[0], W // self.patch_size[1]
        # return x, (H, W)
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False).to(device)
        self.norm = nn.LayerNorm(4 * dim).to(device)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        # B, C, H, W = x.shape
        #
        # x = x.reshape(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)# B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x) # 本来是C 后来变为4C，通过self.reduction(x)变回2C，最终由 C-->2C, 达到分辨率减半，通道数加倍的结果
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], x.shape[1], int(x.shape[2]**0.5), int(x.shape[2]**0.5))

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, embed_dim=96,
                 depths=[2, 2, 6], num_heads=[3, 6, 12],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, **kwargs):

        super().__init__()

        patches_resolution = [img_size // patch_size, img_size // patch_size]
        num_patches = patches_resolution[0] * patches_resolution[1]

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=None)
            self.layers.append(layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

class ConvUpsample(nn.Module):
    def __init__(self, in_chans=256, out_chans = 512, upsample=True):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.patch_size = 4
        if upsample:
            self.convs_level = nn.Sequential(
                nn.Conv2d(self.in_chans, self.out_chans, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(64, self.out_chans),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            )
        else:
            self.convs_level = nn.Sequential(
                nn.Conv2d(self.in_chans, self.out_chans, kernel_size=3, stride=1, padding=1, bias=False),
                nn.MaxPool2d(3, 2, 1),
                nn.GroupNorm(64, self.out_chans),
                nn.GELU(),
            )

    def forward(self, x):
        return self.convs_level(x)

class FeaFusion(nn.Module):
    def __init__(self, image_size, num_classes=11, depth=[1, 1], cnn_pyramid= [96, 192, 384],
                 swin_pyramid=[96, 192, 384], num_heads=[2, 2, 2], mlp_ratio=4., drop=0., attn_drop=0.,
                 norm_layer=nn.LayerNorm, embed_dims=[96, 192, 384], decoder_embed_dim=1024, ):
        super().__init__()
        self.backbone = bockbone_0()
        # self.backbone = vgg_0()
        self.image_size = image_size
        self.patch_size = 4
        self.num_classes = num_classes
        num_patches_0 = (self.image_size // self.patch_size ) ** 2  # default: 3136
        num_patches_2 = (self.image_size // self.patch_size // 4) ** 2  # default: 196

        self.swin_transformer = SwinTransformer(image_size, in_chans = 3).to(device)
        checkpoint = torch.load("/workspace/cls/mae-main/swin_tiny_patch4_window7_224_22k.pth", map_location=torch.device(device))['model']
        unexpected = ["patch_embed.proj.weight", "patch_embed.proj.bias", "patch_embed.norm.weight", "patch_embed.norm.bias",
                     "head.weight", "head.bias", "layers.0.downsample.norm.weight", "layers.0.downsample.norm.bias",
                     "layers.0.downsample.reduction.weight", "layers.1.downsample.norm.weight", "layers.1.downsample.norm.bias",
                     "layers.1.downsample.reduction.weight", "layers.2.downsample.norm.weight", "layers.2.downsample.norm.bias",
                     "layers.2.downsample.reduction.weight", "norm.weight", "norm.bias"]

        self.p1_ch = nn.Conv2d(cnn_pyramid[0], swin_pyramid[0], kernel_size = 1).to(device)
        self.p2_ch = nn.Conv2d(cnn_pyramid[1], swin_pyramid[1], kernel_size=1).to(device)
        self.p3_ch = nn.Conv2d(cnn_pyramid[2], swin_pyramid[2], kernel_size=1).to(device)

        self.patchMerge_0 = PatchMerging((self.image_size // self.patch_size, self.image_size // self.patch_size), swin_pyramid[0])
        self.patchMerge_0.state_dict()['reduction.weight'][:] = checkpoint["layers.0.downsample.reduction.weight"]
        self.patchMerge_0.state_dict()['norm.weight'][:] = checkpoint["layers.0.downsample.norm.weight"]
        self.patchMerge_0.state_dict()['norm.bias'][:] = checkpoint["layers.0.downsample.norm.bias"]

        self.patchMerge_1 = PatchMerging((self.image_size // self.patch_size//2, self.image_size // self.patch_size//2), swin_pyramid[1]).to(device)
        self.patchMerge_1.state_dict()['reduction.weight'][:] = checkpoint["layers.1.downsample.reduction.weight"]
        self.patchMerge_1.state_dict()['norm.weight'][:] = checkpoint["layers.1.downsample.norm.weight"]
        self.patchMerge_1.state_dict()['norm.bias'][:] = checkpoint["layers.1.downsample.norm.bias"]

        for key in list(checkpoint.keys()):
            if key in unexpected or 'layers.3' in key:
                del checkpoint[key]
        self.swin_transformer.load_state_dict(checkpoint)

        self.norm_0 = nn.LayerNorm(swin_pyramid[0]).to(device)
        # self.avgpool_0 = nn.AdaptiveAvgPool1d(1).to(device)

        self.norm_2 = nn.LayerNorm(swin_pyramid[2]).to(device)
        # self.avgpool_2 = nn.AdaptiveAvgPool1d(1).to(device)

        # decoder part
        self.pos_embed_0 = nn.Parameter(torch.zeros(1, num_patches_0 + 1, swin_pyramid[0])).to(device)
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, num_patches_2 + 1, swin_pyramid[2])).to(device)

        self.blocks_0 = nn.Sequential(*[
                Block(embed_dims[0], num_heads[0], mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                      drop=0.1, attn_drop=0.1)
                for i in range(depth[0])]).to(device)

        self.blocks_2 = nn.Sequential(*[
                Block(embed_dims[2], num_heads[1], mlp_ratio, qkv_bias=True, qk_scale=None,
                      norm_layer=norm_layer,drop=0.1, attn_drop=0.1)
                for i in range(depth[1])]).to(device)

        self.ConvUp_0 = ConvUpsample(in_chans=96, out_chans=192, upsample=False).to(device)
        self.ConvUp_2 = ConvUpsample(in_chans=384, out_chans=192, upsample=True).to(device)

        #head classification
        self._fc = nn.Conv2d(embed_dims[1], decoder_embed_dim, kernel_size=1).to(device)
        # self._bn = nn.BatchNorm2d(decoder_embed_dim, eps=1e-5).to(device)
        self._bn = nn.LayerNorm(decoder_embed_dim, eps=1e-5).to(device)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1).to(device)
        self._drop = nn.Dropout(0.1).to(device)
        self.pre_logits = nn.Identity().to(device)

        self.head = nn.Linear(decoder_embed_dim, self.num_classes).to(device) if self.num_classes > 0 else nn.Identity()

        # self.pool = nn.AvgPool2d(decoder_embed_dim, 1)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_encoder(self, x_0, x_1, x_2):
        x_0 = self.p1_ch(x_0)
        x_0 = Rearrange('b c h w -> b (h w) c')(x_0)
        x0 = self.swin_transformer.layers[0](x_0)
        x0_0 = x0 + x_0
        x0_0 = self.norm_0(x0_0)
        x0_1 = self.patchMerge_0(x0_0)

        x0_1 = Rearrange('b c h w -> b (h w) c')(x0_1)
        x1 = self.swin_transformer.layers[1](x0_1)
        x_1 = self.p2_ch(x_1)
        x_1 = Rearrange('b c h w -> b (h w) c')(x_1)
        x1_1 = x1 + x_1
        x1_1 = self.patchMerge_1(x1_1)

        x1_1 = Rearrange('b c h w -> b (h w) c')(x1_1)
        x2 = self.swin_transformer.layers[2](x1_1)
        x_2 = self.p3_ch(x_2)
        x_2 = Rearrange('b c h w -> b (h w) c')(x_2)
        x2_1 = x2 + x_2
        x2_1 = self.norm_2(x2_1)

        return x0_0, x2_1

    def forward_decoder(self, x0, x2):
        # x0 = x0 + self.pos_embed_0[:, 1:, :]
        # x2 = x2 + self.pos_embed_2[:, 1:, :]
        # x0 = self.blocks_0(x0)
        # x2 = self.blocks_2(x2)
        #
        # x0 = self.norm_0(x0)
        # x2 = self.norm_2(x2)

        x0 = Rearrange('b (h w) c ->b c h w',
                       h=(self.image_size // self.patch_size), w=(self.image_size // self.patch_size))(x0)
        x2 = Rearrange('b (h w) c ->b c h w',
                       h=(self.image_size // self.patch_size//4), w=(self.image_size // self.patch_size//4))(x2)
        x0_conv = self.ConvUp_0(x0)
        x2_conv = self.ConvUp_2(x2)
        x = x0_conv + x2_conv

        return x

    def forward_head(self, x):
        x = self._fc(x)
        x = Rearrange('b c h w -> b (h w) c')(x)
        x = self._bn(x)
        x = Rearrange('b (h w) d -> b d h w', h=(self.image_size // self.patch_size // 2),
                             w=(self.image_size // self.patch_size // 2))(x)
        x = self._avg_pooling(x).flatten(start_dim=1)
        x = self._drop(x)
        x = self.pre_logits(x)
        x = self.head(x)
        return x

    def forward(self, x):
        x0, x1, x2 = self.backbone(x)
        x_0, x_2 = self.forward_encoder(x0, x1, x2)
        x = self.forward_decoder(x_0, x_2)
        x = self.forward_head(x)

        return x

    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)

def FeaFusion_0():

    return FeaFusion(224, num_classes=10, embed_dims=[96, 192, 384], decoder_embed_dim=192)
    #RESNET
    # return FeaFusion(224, num_classes=10, embed_dims=[512, 1024, 2048], decoder_embed_dim=1024)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    img = torch.randn(1, 3, 224, 224)

    net = FeaFusion_0()
    out = net(img)
    print(out.shape, count_parameters(net))
    # print(count_parameters(net))