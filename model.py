import segmentation_models_pytorch as smp

def get_model(encoder="resnet34", weights="imagenet", classes=1):
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=4,
        classes=classes,
        activation=None,
    )
    