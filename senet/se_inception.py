from se_module import SELayer
from torch import nn
from torchvision.models.inception import Inception3

import torch

class SEInception3(nn.Module):
    def __init__(self, num_classes, aux_logits=True, transform_input=False):
        super(SEInception3, self).__init__()
        model = Inception3(num_classes=num_classes, aux_logits=aux_logits,
                           transform_input=transform_input)
        model.Mixed_5b.add_module("SELayer", SELayer(192))
        model.Mixed_5c.add_module("SELayer", SELayer(256))
        model.Mixed_5d.add_module("SELayer", SELayer(288))
        model.Mixed_6a.add_module("SELayer", SELayer(288))
        model.Mixed_6b.add_module("SELayer", SELayer(768))
        model.Mixed_6c.add_module("SELayer", SELayer(768))
        model.Mixed_6d.add_module("SELayer", SELayer(768))
        model.Mixed_6e.add_module("SELayer", SELayer(768))
        if aux_logits:
            model.AuxLogits.add_module("SELayer", SELayer(768))
        model.Mixed_7a.add_module("SELayer", SELayer(768))
        model.Mixed_7b.add_module("SELayer", SELayer(1280))
        model.Mixed_7c.add_module("SELayer", SELayer(2048))

        self.model = model

    def forward(self, x):
        _, _, h, w = x.size()
        if (h, w) != (299, 299):
            raise ValueError("input size must be (299, 299)")

        return self.model(x)


def se_inception_v3(**kwargs):
    return SEInception3(**kwargs)


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


if __name__ == '__main__':

    net = se_inception_v3(num_classes=1000)
    print(net)
    print(get_n_params(net))
    x = torch.rand(2,3,299,299)
    y = net(x)
    print(y[0].size())