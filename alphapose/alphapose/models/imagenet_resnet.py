import torchvision.models as tm


def imagenet_resnet(num_layers):
    """Torchvision ResNet with ImageNet weights, on old and new torchvision.

    torchvision 0.13 replaced ``pretrained=True`` with ``weights=<enum>`` and
    warns that ``pretrained`` may be removed; older releases only know
    ``pretrained``. IMAGENET1K_V1 is the weight set ``pretrained=True`` loads.
    """
    build = getattr(tm, f"resnet{num_layers}")
    weights = getattr(tm, f"ResNet{num_layers}_Weights", None)
    if weights is None:  # torchvision < 0.13
        return build(pretrained=True)
    return build(weights=weights.IMAGENET1K_V1)
