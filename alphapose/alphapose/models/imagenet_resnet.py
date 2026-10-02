import torchvision.models as tm


def imagenet_resnet(num_layers, pretrained=True):
    """Torchvision ResNet with ImageNet weights, on old and new torchvision.

    ``pretrained=False`` (model config ``BACKBONE_PRETRAINED: false``) skips the ImageNet
    download; use it when a full checkpoint is loaded afterwards.

    torchvision 0.13 replaced ``pretrained=True`` with ``weights=<enum>`` and
    warns that ``pretrained`` may be removed; older releases only know
    ``pretrained``. IMAGENET1K_V1 is the weight set ``pretrained=True`` loads.
    """
    build = getattr(tm, f"resnet{num_layers}")
    weights = getattr(tm, f"ResNet{num_layers}_Weights", None)
    if not pretrained:
        # Random init: callers that load a full checkpoint afterwards do not need the download.
        return build(weights=None) if weights is not None else build(pretrained=False)
    if weights is None:  # torchvision < 0.13
        return build(pretrained=True)
    return build(weights=weights.IMAGENET1K_V1)
