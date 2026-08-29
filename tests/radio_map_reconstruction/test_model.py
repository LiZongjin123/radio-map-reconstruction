import pytest
from torch import zeros

from radio_map_reconstruction.model import ResUnet


@pytest.mark.parametrize("in_channels", (2, 4))
def test_res_unet_accepts_the_configured_input_channels(in_channels):
    model = ResUnet(in_channels=in_channels, base_channels=1)

    output = model(zeros((1, in_channels, 16, 16)))

    assert output.shape == (1, 1, 16, 16)


def test_res_unet_rejects_inputs_that_do_not_match_its_channel_contract():
    model = ResUnet(in_channels=2, base_channels=1)

    with pytest.raises(
        ValueError,
        match="expected 2 input channels, but received 4",
    ):
        model(zeros((1, 4, 16, 16)))
