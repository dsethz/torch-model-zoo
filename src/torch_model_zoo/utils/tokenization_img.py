#######################################################################################################################
# This script contains image tokenization approaches.                                                                 #
# Author:               Daniel Schirmacher                                                                            #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                               #
# Python Version:       3.11.7                                                                                        #
# PyTorch Version:      2.3.0                                                                                         #
# PyTorch Lightning Version: 1.5.9                                                                                    #
#######################################################################################################################


# class DoubleConv(nn.Module):
#     """
#     This class does (convolution => [BN] => ReLU) * 2.
#     """
#
#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         """
#         Constructor.
#
#
#         Parameter
#         ---------
#
#         in_channels: int
#             Input filters.
#
#         out_channels: int
#             Output filters.
#
#         mid_channels: int
#             intermediate filters.
#
#
#         Return
#         ------
#
#         -
#         """
#         super().__init__()
#
#         if mid_channels is None:
#             mid_channels = out_channels
#
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(mid_channels),
#             nn.LeakyReLU(negative_slope=slope, inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(inplace=True),
#         )
#
#         self.double_conv.apply(init_weights)
#
#     def forward(self, x):
#         return self.double_conv(x)
