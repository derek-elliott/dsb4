import torch

from skimage import transform


class Rescale(object):
    """
    Rescale the image in a sample to the given size.

    Args:
        output_size (tuple): Desired output size. Should be of the form: (height, width)
    """

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        image, masks, combined_mask = sample['image'], sample['masks'], sample['combined_mask']

        img = transform.resize(image, self.output_size)
        combined_msk = transform.resize(combined_mask, self.output_size)

        msks = []
        for mask in masks:
            msks.append(transform.resize(mask, self.output_size))

        return {'image': img, 'combined_mask': combined_msk, 'masks': msks}


class ToTensor(object):
    """
    Convert images to Tensors.
    """

    def __call__(self, sample):
        image, masks, combined_mask = sample['image'], sample['masks'], sample['combined_mask']
        img = image.transpose((2, 0, 1))
        msks = []
        for mask in masks:
            msks.append(torch.from_numpy(mask))

        return {'image': torch.from_numpy(img).float(), 'combined_mask': torch.from_numpy(combined_mask), 'masks': msks}
