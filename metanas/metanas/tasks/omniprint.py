import os
import json
import glob
import h5py
from PIL import Image, ImageOps

from torchvision.datasets.utils import list_dir, download_url

from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from torchmeta.datasets.utils import get_asset
from torchmeta.datasets.helpers import helper_with_default


def omniPrint(folder, shots, ways, shuffle=True, test_shots=None, seed=None, **kwargs):
    return helper_with_default(OmniPrint, folder, shots, ways,
                               shuffle=shuffle, test_shots=test_shots,
                               seed=seed, **kwargs)


class OmniPrint(CombinationMetaDataset):

    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 print_split="meta1",
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = OmniPrintClassDataset(root, meta_train=meta_train,
                                        meta_val=meta_val, meta_test=meta_test,
                                        print_split=print_split, transform=transform,
                                        meta_split=meta_split,
                                        class_augmentations=class_augmentations,
                                        download=download)
        super(OmniPrint, self).__init__(dataset, num_classes_per_task,
                                        target_transform=target_transform,
                                        dataset_transform=dataset_transform)


class OmniPrintClassDataset(ClassDataset):
    # TODO: Add which version .e.g 1-5
    folder = 'omniprint'
    # TODO: Adjust correct link
    # gdrive_id = '1UkbNDPPOiEQWl03hbuJfBXYVn96sNS74'
    zip_filename = 'omniprint.zip'

    filename = '{0}_data.hdf5'
    filename_labels = '{0}_{1}_labels.json'

    image_folder = 'omniprint'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, print_split="meta1", transform=None,
                 class_augmentations=None, download=False):
        super(OmniPrintClassDataset, self).__init__(meta_train=meta_train,
                                                    meta_val=meta_val,
                                                    meta_test=meta_test,
                                                    meta_split=meta_split,
                                                    class_augmentations=class_augmentations)

        self.root = os.path.join(os.path.expanduser(
            root), self.folder)
        self.data_root = os.path.join(os.path.expanduser(
            root), self.folder, print_split)

        self.print_split = print_split
        self.transform = transform

        self.split_filename = os.path.join(self.root, self.filename)
        self.split_filename_labels = os.path.join(
            self.root,
            self.filename_labels.format(print_split, self.meta_split))

        self._data = None
        self._labels = None

        # TODO: Fix download gdrive link
        if download:
            self.download()

        # if not self._check_integrity():
        #     raise RuntimeError('OmniPrint integrity check failed')
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        character_name = '/'.join(self.labels[index % self.num_classes])
        data = self.data[character_name]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return OmniPrintDataset(index, data, character_name,
                                transform=transform, target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data = h5py.File(self.split_filename, 'r')
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels

    def _check_integrity(self):
        return (os.path.isfile(self.split_filename)
                and os.path.isfile(self.split_filename_labels))

    def close(self):
        if self._data is not None:
            self._data.close()
            self._data = None

    def download(self):
        import zipfile
        import shutil

        if self._check_integrity():
            return

        # for name in self.zips_md5:
        #     zip_filename = '{0}.zip'.format(name)
        #     filename = os.path.join(self.root, zip_filename)
        #     if os.path.isfile(filename):
        #         continue

        #     # url = '{0}/{1}'.format(self.download_url_prefix, zip_filename)
        #     # download_url(url, self.root, zip_filename, self.zips_md5[name])

        #     with zipfile.ZipFile(filename, 'r') as f:
        #         f.extractall(self.root)

        filename = os.path.join(self.root, self.filename)
        with h5py.File(filename, 'w') as f:
            for name in self.zips_md5:
                group = f.create_group(name)

                alphabets = list_dir(os.path.join(self.root, name))
                characters = [(name, alphabet, character) for alphabet in alphabets
                              for character in list_dir(os.path.join(self.root, name, alphabet))]

                # split = 'train' if name == 'images_background' else 'test'
                labels_filename = os.path.join(self.root,
                                               self.filename_labels.format(self.print_split, self.meta_split))
                with open(labels_filename, 'w') as f_labels:
                    labels = sorted(characters)
                    json.dump(labels, f_labels)

                for _, alphabet, character in characters:
                    filenames = glob.glob(os.path.join(self.root, name,
                                                       alphabet, character, '*.png'))
                    dataset = group.create_dataset('{0}/{1}'.format(alphabet,
                                                                    character), (len(filenames), 105, 105), dtype='uint8')

                    for i, char_filename in enumerate(filenames):
                        image = Image.open(
                            char_filename, mode='r').convert('L')
                        dataset[i] = ImageOps.invert(image)

                shutil.rmtree(os.path.join(self.root, name))

        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, self.filename_labels.format(
                'vinyals_', split))
            data = get_asset(
                self.folder, '{0}.json'.format(split), dtype='json')

            with open(filename, 'w') as f:
                labels = sorted([('images_{0}'.format(name), alphabet, character)
                                 for (name, alphabets) in data.items()
                                 for (alphabet, characters) in alphabets.items()
                                 for character in characters])
                json.dump(labels, f)


class OmniPrintDataset(Dataset):
    def __init__(self, index, data, character_name,
                 transform=None, target_transform=None):
        super(OmniPrintDataset, self).__init__(index, transform=transform,
                                               target_transform=target_transform)
        self.data = data
        self.character_name = character_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        target = self.character_name

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)


    # def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
    #              meta_split=None, transform=None, class_augmentations=None,
        # 			 download=False):
ds = OmniPrintClassDataset(
    "/home/rob/Git/meta-fsl-nas/data/", meta_train=True, download=False)


dataloader = BatchMetaDataLoader(
    ds, batch_size=1, num_workers=1, shuffle=True
)
batch = next(iter(dataloader))
print(batch)
