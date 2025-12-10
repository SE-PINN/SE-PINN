from itertools import permutations

import pytest
import torch.nn as nn

from sepinn.hub_layer import HubLayer


class TestInitialization:
    def test_class(self):
        hub_layer = HubLayer(50, 50, 0, 0)

        assert isinstance(hub_layer, HubLayer)

    def test_size_in(self):
        hub_layer = HubLayer(50, 50, 0, 0)

        assert hub_layer.size_in == 50

    def test_size_out(self):
        hub_layer = HubLayer(50, 50, 0, 0)

        assert hub_layer.size_out == 50

    def test_even(self):
        hub_layer = HubLayer(50, 50, 0, 0)

        assert hub_layer.even == 0

    def test_odd(self):
        hub_layer = HubLayer(50, 50, 0, 0)

        assert hub_layer.odd == 0

    def test_weights(self):
        hub_layer = HubLayer(50, 50, 0, 0)

        assert isinstance(hub_layer.weights, nn.Parameter)

    def test_bias(self):
        hub_layer = HubLayer(50, 50, 0, 0)

        assert isinstance(hub_layer.bias, nn.Parameter)


def test_flip_sym():
    for even, odd in permutations((1, 0)):
        hub_layer = HubLayer(50, 50, even, odd)

        hub_layer.flip_sym()

        assert hub_layer.even == odd
        assert hub_layer.odd == even
