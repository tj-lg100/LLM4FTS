import torch
import torch.nn as nn
import pywt

class WaveletTransform(nn.Module):
    def __init__(self, in_channels, wavelet='db1'):
        super(WaveletTransform, self).__init__()
        self.in_channels = in_channels

        wavelet = pywt.Wavelet(wavelet)
        self.low_pass_filter = nn.Parameter(
            torch.tensor(wavelet.dec_lo, dtype=torch.float32), requires_grad=True)
        self.high_pass_filter = nn.Parameter(
            torch.tensor(wavelet.dec_hi, dtype=torch.float32), requires_grad=True)
        
        self.low_pass_reconstruct = torch.tensor(wavelet.rec_lo, dtype=torch.float32)
        self.high_pass_reconstruct = torch.tensor(wavelet.rec_hi, dtype=torch.float32)

        self.conv_low = nn.Conv1d(in_channels, in_channels, kernel_size=len(self.low_pass_filter), stride=2, padding=len(self.low_pass_filter) // 2, groups=in_channels, bias=True)
        self.conv_high = nn.Conv1d(in_channels, in_channels, kernel_size=len(self.high_pass_filter), stride=2, padding=len(self.high_pass_filter) // 2, groups=in_channels, bias=True)

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        self.conv_low.weight.data = self.low_pass_filter.view(1, 1, -1).repeat(self.in_channels, 1, 1)
        self.conv_high.weight.data = self.high_pass_filter.view(1, 1, -1).repeat(self.in_channels, 1, 1)
    
    def orthogonalize_filters(self):
        with torch.no_grad():
            for c in range(self.conv_low.weight.data.shape[0]):  # 遍历每个通道
                low_c = self.conv_low.weight.data[c, 0, :]  # [8]
                high_c = self.conv_high.weight.data[c, 0, :]

                # 正交化
                dot_product = torch.dot(low_c, high_c)
                norm_high = torch.norm(high_c) ** 2 + 1e-6
                norm_low = torch.norm(low_c) ** 2 + 1e-6

                ortho_low = low_c - dot_product * high_c / norm_high
                ortho_high = high_c - dot_product * low_c / norm_low

                # 写回权重
                self.conv_low.weight.data[c, 0, :] = ortho_low
                self.conv_high.weight.data[c, 0, :] = ortho_high

    def forward(self, x):
        B, M, L = x.shape

        x_low = self.conv_low(x)
        x_high = self.conv_high(x)

        self.orthogonalize_filters()
       
        return x_low, x_high

    def inverse_forward(self, x_low, x_high):
        B, M, L = x_low.shape
        padded_len = L * 2

        # Upsample by inserting zeros
        x_low_upsampled = torch.zeros((B, M, padded_len), device=x_low.device)
        x_high_upsampled = torch.zeros((B, M, padded_len), device=x_high.device)

        x_low_upsampled[..., ::2] = x_low
        x_high_upsampled[..., ::2] = x_high

        conv_low_reconstruct = nn.Conv1d(M, M, kernel_size=len(self.low_pass_reconstruct), padding=len(self.low_pass_reconstruct) // 2, groups=M, bias=False)
        conv_high_reconstruct = nn.Conv1d(M, M, kernel_size=len(self.high_pass_reconstruct), padding=len(self.high_pass_reconstruct) // 2, groups=M, bias=False)

        conv_low_reconstruct.weight.data = self.low_pass_reconstruct.view(1, 1, -1).repeat(M, 1, 1)
        conv_high_reconstruct.weight.data = self.high_pass_reconstruct.view(1, 1, -1).repeat(M, 1, 1)

        x_low_filtered = conv_low_reconstruct(x_low_upsampled)
        x_high_filtered = conv_high_reconstruct(x_high_upsampled)

        output = x_low_filtered + x_high_filtered

        return output