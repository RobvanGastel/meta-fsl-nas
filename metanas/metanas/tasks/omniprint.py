from torchmeta.utils.data import BatchMetaDataLoader
import os
import json
import h5py
import numpy as np
from PIL import Image

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from torchmeta.datasets.utils import download_file_from_google_drive
from torchmeta.datasets.helpers import helper_with_default


def omniprint(folder, shots, ways, shuffle=True, test_shots=None,
              seed=None, **kwargs):
    return helper_with_default(OmniPrint, folder, shots, ways,
                               shuffle=shuffle, test_shots=test_shots,
                               seed=seed, **kwargs)


class OmniPrint(CombinationMetaDataset):
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False,
                 print_split='meta1',  # Addition for the OmniPrint dataset
                 ):
        dataset = OmniPrintClassDataset(
            root, meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test,
            print_split=print_split, transform=transform,
            meta_split=meta_split,
            class_augmentations=class_augmentations,
            download=download)
        super(OmniPrint, self).__init__(
            dataset, num_classes_per_task,
            target_transform=target_transform,
            dataset_transform=dataset_transform)


class OmniPrintClassDataset(ClassDataset):
    gdrive_id = '1JBXYMTsdlm8RaEBPqrJbDRzs3hJ4q_gH'
    folder = 'omniprint'

    zip_filename = '{0}.zip'
    filename = '{0}_{1}_data.hdf5'
    filename_labels = '{0}_{1}_labels.json'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, print_split="meta1", transform=None,
                 class_augmentations=None, download=False):
        super(OmniPrintClassDataset, self).__init__(
            meta_train=meta_train,
            meta_val=meta_val,
            meta_test=meta_test,
            meta_split=meta_split,
            class_augmentations=class_augmentations)

        self.root = os.path.join(os.path.expanduser(
            root), self.folder)
        self.print_split = print_split
        self.transform = transform

        self.split_filename = os.path.join(
            self.root,
            self.filename.format(print_split, self.meta_split))
        self.split_filename_labels = os.path.join(
            self.root,
            self.filename_labels.format(print_split, self.meta_split))

        self._data = None
        self._labels = None

        if download:
            self.download()

        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        character_name = '/'.join(self.labels[index % self.num_classes])
        data = self.data[character_name]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return OmniPrintDataset(
            index, data, character_name,
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
        import glob
        from tqdm import tqdm

        zip_foldername = os.path.join(
            self.root, self.zip_filename.format(self.folder))
        # Download the datasets
        if not os.path.isfile(zip_foldername):
            download_file_from_google_drive(
                self.gdrive_id, self.root,
                self.zip_filename.format(self.folder))

        # Unzip the dataset
        if not os.path.isdir(zip_foldername):
            with zipfile.ZipFile(zip_foldername) as f:
                for member in tqdm(f.infolist(), desc='Extracting '):
                    try:
                        f.extract(member, self.root)
                    except zipfile.BadZipFile:
                        print('Error: Zipfile is corrupted')

        for print_split in ['meta1', 'meta2', 'meta3', 'meta4', 'meta5']:
            for split in tqdm(['train', 'val', 'test'], desc=print_split):
                filename_labels = os.path.join(
                    self.root, self.filename_labels.format(print_split, split))

                with open(filename_labels, 'r') as f:
                    labels = json.load(f)

                filename = os.path.join(
                    self.root, self.filename.format(print_split, split))

                with h5py.File(filename, 'w') as f:
                    group = f.create_group(print_split)
                    for _, alphabet, character in labels:
                        filenames = glob.glob(
                            os.path.join(
                                self.root, print_split,
                                alphabet, character, '*.png'))
                        dataset = group.create_dataset('{0}/{1}'.format(
                            alphabet, character),
                            (len(filenames), 32, 32),
                            dtype='uint8')

                        for i, char_filename in enumerate(filenames):
                            image = Image.open(
                                char_filename, mode='r').convert('L')
                            dataset[i] = image

            shutil.rmtree(os.path.join(self.root, print_split))


class OmniPrintDataset(Dataset):
    def __init__(self, index, data, character_name,
                 transform=None, target_transform=None):
        super(OmniPrintDataset, self).__init__(
            index, transform=transform,
            target_transform=target_transform)
        self.data = data
        self.character_name = character_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = np.asarray(self.data[index])
        target = self.character_name

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return (image, target)
