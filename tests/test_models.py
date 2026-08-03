import torch

from banknote_classifier.models import VGG16

NUM_CLASSES = 9
IMAGE_SIZE = 224


def test_vgg16_instantiation():
    model = VGG16(num_classes=NUM_CLASSES)
    assert isinstance(model, torch.nn.Module)


def test_vgg16_output_shape():
    model = VGG16(num_classes=NUM_CLASSES)
    model.eval()
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, NUM_CLASSES + 1)


def test_vgg16_output_is_tensor():
    model = VGG16(num_classes=NUM_CLASSES)
    model.eval()
    dummy_input = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)

    with torch.no_grad():
        output = model(dummy_input)

    assert isinstance(output, torch.Tensor)
