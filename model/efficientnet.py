from efficientnet_pytorch import EfficientNet


def get_model(name, num_classes):
    model = EfficientNet.from_pretrained(name, num_classes)
    return model