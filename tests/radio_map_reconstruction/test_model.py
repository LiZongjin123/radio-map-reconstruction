from pytest import raises
from torch import zeros

from radio_map_reconstruction.model import ResUnet


def test_create_res_unet():
    model = ResUnet()
    assert model is not None


def test_configured_two_channel_res_unet_runs_a_forward_pass():
    model = ResUnet(in_channels=2, base_channels=1)

    output = model(zeros(1, 2, 16, 16))

    assert output.shape == (1, 1, 16, 16)


def test_default_res_unet_retains_four_channel_contract():
    model = ResUnet(base_channels=1)

    assert model(zeros(1, 4, 16, 16)).shape == (1, 1, 16, 16)
    with raises(ValueError, match="通道数不等于4"):
        model(zeros(1, 2, 16, 16))
