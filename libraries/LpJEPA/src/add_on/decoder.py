import torch.nn as nn


class ImageConvDecoder(nn.Module):
    def __init__(self, d_in, linear_depth: int | None = None, in_w: int = 8, in_h: int = 8):
        super().__init__()
        self.linear_depth = 64 if linear_depth is None else linear_depth
        self.in_w = in_w
        self.in_h = in_h
        self.fc = nn.Linear(d_in, self.linear_depth * in_w * in_h)
        self.decoder = nn.Sequential(
            self._conv_block(c_in=self.linear_depth, c_out=256, kernel_size=4, stride=2, padding=1),
            self._conv_block(c_in=256, c_out=128, kernel_size=4, stride=2, padding=1),
            self._conv_block(c_in=128, c_out=64, kernel_size=4, stride=2, padding=1),
            self._conv_block(c_in=64, c_out=32, kernel_size=4, stride=2, padding=1),
            self._conv_block(c_in=32, c_out=16, kernel_size=4, stride=2, padding=1),
            self._conv_block(c_in=16, c_out=1, kernel_size=4, stride=2, padding=1)
        )
        self.act = nn.Sigmoid()

    @staticmethod
    def _conv_block(c_in, c_out, padding, kernel_size, stride):
        return nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, kernel_size, stride, padding),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.fc(x).view(-1, self.linear_depth, self.in_w, self.in_h)
        return self.act(self.decoder(x))


class ImageDiffusionDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.Identity(x)
