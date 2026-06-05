import segmentation_models_pytorch as smp


def get_model():

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=83,
        classes=1
    )

    return model