import torch
import torch.nn as nn

from .common import BSConv, ESA

class DilationConvGroup(nn.Module):

    def __init__(self, num_feat):
        super(DilationConvGroup, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=1)
        )

    def forward(self, x):
        return self.convs(x)

class BSConvGroup(nn.Module):

    def __init__(self, num_feat):
        super(BSConvGroup, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, kernel_size=1),
            nn.ReLU(inplace=True),
            BSConv(num_feat, num_feat, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            BSConv(num_feat, num_feat, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            BSConv(num_feat, num_feat, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=1)
        )

    def forward(self, x):
        return self.convs(x)

class LFPB(nn.Module):

    def __init__(self, num_feat):
        super(LFPB, self).__init__()
        self.dilation_group = DilationConvGroup(num_feat)
        self.bsconv_group = BSConvGroup(num_feat)
        self.esa = ESA(num_feat)
        self.bconv_final = BSConv(num_feat, num_feat, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        dilated_feat = self.dilation_group(x)
        bsconv_feat = self.bsconv_group(x)

        fused_feat = dilated_feat + bsconv_feat

        attended_feat = self.esa(fused_feat)

        res = attended_feat + identity
        out = self.bconv_final(res)

        return out

class LFPN(nn.Module):

    def __init__(self, scale=2, num_feat=56, num_blocks=6):
        super(LFPN, self).__init__()
        self.scale = scale

        self.head = nn.Conv2d(3, num_feat, kernel_size=3, padding=1)

        self.body = nn.ModuleList([LFPB(num_feat) for _ in range(num_blocks)])

        self.fusion = nn.Sequential(
            nn.Conv2d(num_feat * num_blocks, num_feat, kernel_size=1, padding=0),
            nn.Sigmoid(),
            nn.ReLU(inplace=False),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1)
        )

        self.tail = nn.Sequential(
            nn.Conv2d(num_feat, 3 * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):

        x = self.head(x)

        identity = x
        features = []
        for block in self.body:
            x = block(x)
            features.append(x)

        concatenated_features = torch.cat(features, 1)
        fused_features = self.fusion(concatenated_features)

        fused_features = fused_features + identity

        out = self.tail(fused_features)
        return out

if __name__ == "__main__":

    UPSCALE_FACTOR = 2
    NUM_FEATURES = 56
    NUM_BLOCKS = 6

    model = LFPN(scale=UPSCALE_FACTOR, num_feat=NUM_FEATURES, num_blocks=NUM_BLOCKS)
    print(f"LFPN model created successfully.")
    print(f"Scale: {UPSCALE_FACTOR}, Features: {NUM_FEATURES}, Blocks: {NUM_BLOCKS}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params / 1e3:.1f}K")

    dummy_input = torch.randn(1, 3, 64, 64)
    print(f"\nInput tensor shape: {dummy_input.shape}")

    try:
        with torch.no_grad():
            model.eval()
            output = model(dummy_input)
        print(f"Forward pass successful!")
        print(f"Output tensor shape: {output.shape}")

        expected_h = dummy_input.shape[2] * UPSCALE_FACTOR
        expected_w = dummy_input.shape[3] * UPSCALE_FACTOR
        assert output.shape == (1, 3, expected_h, expected_w), "Output shape is incorrect!"
        print("Output shape is correct.")

    except Exception as e:
        print(f"An error occurred during the forward pass: {e}")
