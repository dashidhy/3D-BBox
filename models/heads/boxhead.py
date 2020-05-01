import torch
from torch import nn

__all__ = [
    'BoxHead'
]

def fbr_layer(in_size, out_size, bias=False):
    return nn.Sequential(
        nn.Linear(in_size, out_size, bias=bias),
        nn.BatchNorm1d(out_size),
        nn.ReLU(inplace=True)
    )

class BoxHead(nn.Module):

    def __init__(
            self,
            in_size=512*7*7,
            num_bins=2,
            dim_reg_hide_sizes=[512],
            bin_conf_hide_sizes=[256],
            bin_reg_hide_sizes=[256],
            cos_sin_encode=False,
            init_weights=True
        ):
        super(BoxHead, self).__init__()

        self.in_size = in_size
        self.num_bins = num_bins

        self.dim_reg_layers = self._make_fc_layers(dim_reg_hide_sizes, 3)
        self.bin_conf_layers = self._make_fc_layers(bin_conf_hide_sizes, num_bins)
        self.cos_sin_encode = cos_sin_encode
        bin_reg_out_size = num_bins * 2 if self.cos_sin_encode else num_bins
        self.bin_reg_layers = self._make_fc_layers(bin_reg_hide_sizes, bin_reg_out_size)

        if init_weights:
            self.init_weights()
    
    def _make_fc_layers(self, hidden_sizes, out_size):
        fc_layers = []
        pre_size = self.in_size
        for hidden_size in hidden_sizes:
            fc_layers.append(fbr_layer(pre_size, hidden_size))
            pre_size = hidden_size
        fc_layers.append(nn.Linear(pre_size, out_size))
        return nn.Sequential(*fc_layers)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Input:
            x: Tensor(N, self.in_size), flattened feature map
               from backbone net

        Return:
            dim_reg: Tensor(N, 3), each row (dh, dw, dl)
            bin_conf: Tensor(N, num_bins), bin confidence scores
            bin_reg: Tensor(N, num_bins, 2), 
                     each bin (cos_encode, sin_encode),
                     should be normalized for loss or
                     use torch.atan2 to get the real angle
        """

        # forward to fc layers
        dim_reg = self.dim_reg_layers(x)
        bin_conf = self.bin_conf_layers(x)
        bin_reg = self.bin_reg_layers(x)
        if self.cos_sin_encode:
            bin_reg = bin_reg.view(-1, self.num_bins, 2)

        return dim_reg, bin_conf, bin_reg