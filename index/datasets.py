import numpy as np
import torch
import torch.utils.data as data
import os

class EmbDataset(data.Dataset):

    def __init__(self,data_path):

        self.data_path = data_path
        # self.embeddings = np.fromfile(data_path, dtype=np.float32).reshape(16859,-1)
        self.embeddings = np.load(data_path)
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)
    
class EmbDatasetAll(data.Dataset):

    def __init__(self, args):

        self.datasets = args.datasets.split(',')
        embeddings = []
        self.dataset_count = []
        for dataset in self.datasets:
            print(dataset)
            embedding_path = os.path.join(args.data_root, dataset, f'{dataset}{args.embedding_file}')
            embedding = np.load(embedding_path)
            embeddings.append(embedding)
            self.dataset_count.append(embedding.shape[0])
            
        self.embeddings = np.concatenate(embeddings)
        self.dim = self.embeddings.shape[-1]
        
        print(self.dataset_count)
        print(self.embeddings.shape[0])

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)
    
class EmbDatasetOne(data.Dataset):

    def __init__(self, args, dataset):


        print(dataset)
        embedding_path = os.path.join(args.data_root, dataset, f'{dataset}{args.embedding_file}')
        self.embedding = np.load(embedding_path)

        self.dim = self.embedding.shape[-1]
        
        self.data_count = self.embedding.shape[0]

        print(self.embedding.shape)

    def __getitem__(self, index):
        emb = self.embedding[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embedding)

class DualEmbDataset(data.Dataset):

    def __init__(self, args, dataset):
        """
        Load dual-modal (text + image) embeddings from ONE dataset.

        Directory structure:
            args.data_root/
                dataset/
                    dataset{args.text_embedding_file}
                    dataset{args.image_embedding_file}
        """

        self.dataset = dataset
        self.data_root = args.data_root
        
        # build paths by concatenation
        self.text_embedding_path = os.path.join(
            self.data_root,
            dataset,
            f"{dataset}{args.text_embedding_file}"
        )
        self.image_embedding_path = os.path.join(
            self.data_root,
            dataset,
            f"{dataset}{args.image_embedding_file}"
        )

        # load embeddings
        self.text_embeddings = np.load(self.text_embedding_path)
        self.img_embeddings = np.load(self.image_embedding_path)
        self.data_count = self.text_embeddings.shape[0]
        
        assert len(self.text_embeddings) == len(self.img_embeddings), \
            (
                f"[{dataset}] Text/Image length mismatch: "
                f"{len(self.text_embeddings)} vs {len(self.img_embeddings)}"
            )

        self._text_dim = self.text_embeddings.shape[-1]
        self._img_dim = self.img_embeddings.shape[-1]

        print("====================================")
        print(f"DualEmbDataset summary ({dataset}):")
        print(f"  Text path: {self.text_embedding_path}")
        print(f"  Image path: {self.image_embedding_path}")
        print(f"  Text shape: {self.text_embeddings.shape}")
        print(f"  Image shape: {self.img_embeddings.shape}")
        print(f"  Total samples: {len(self.text_embeddings)}")
        print("====================================")

    def __getitem__(self, index):
        text_emb = self.text_embeddings[index]
        img_emb = self.img_embeddings[index]

        text_tensor = torch.FloatTensor(text_emb)
        img_tensor = torch.FloatTensor(img_emb)

        return text_tensor, img_tensor, index

    def __len__(self):
        return len(self.text_embeddings)

    @property
    def text_dim(self):
        return self._text_dim

    @property
    def img_dim(self):
        return self._img_dim

class DualEmbAllDataset(data.Dataset):

    def __init__(self, args):
        """
        Load dual-modal (text + image) embeddings from multiple datasets
        and concatenate them into a single dataset.

        Expected file structure:
        args.data_root/
            dataset1/
                dataset1{text_embedding_file}
                dataset1{image_embedding_file}
            dataset2/
                dataset2{text_embedding_file}
                dataset2{image_embedding_file}
            ...
        """

        self.datasets = args.datasets.split(',')

        text_embeddings_all = []
        img_embeddings_all = []

        self.dataset_count = []

        for dataset in self.datasets:
            print(f"Loading dataset: {dataset}")

            text_path = os.path.join(
                args.data_root,
                dataset,
                f"{dataset}{args.text_embedding_file}"
            )
            img_path = os.path.join(
                args.data_root,
                dataset,
                f"{dataset}{args.image_embedding_file}"
            )

            text_emb = np.load(text_path)
            img_emb = np.load(img_path)

            assert len(text_emb) == len(img_emb), \
                f"[{dataset}] Text/Image length mismatch: {len(text_emb)} vs {len(img_emb)}"

            text_embeddings_all.append(text_emb)
            img_embeddings_all.append(img_emb)

            self.dataset_count.append(len(text_emb))

            print(f"  Text shape: {text_emb.shape}")
            print(f"  Image shape: {img_emb.shape}")

        # concatenate along item dimension
        self.text_embeddings = np.concatenate(text_embeddings_all, axis=0)
        self.img_embeddings = np.concatenate(img_embeddings_all, axis=0)

        self._text_dim = self.text_embeddings.shape[-1]
        self._img_dim = self.img_embeddings.shape[-1]

        print("====================================")
        print("DualEmbAllDataset summary:")
        print(f"  Datasets: {self.datasets}")
        print(f"  Per-dataset counts: {self.dataset_count}")
        print(f"  Total items: {len(self.text_embeddings)}")
        print(f"  Text dim: {self._text_dim}")
        print(f"  Image dim: {self._img_dim}")
        print("====================================")

    def __getitem__(self, index):
        text_emb = self.text_embeddings[index]
        img_emb = self.img_embeddings[index]

        text_tensor = torch.FloatTensor(text_emb)
        img_tensor = torch.FloatTensor(img_emb)

        return text_tensor, img_tensor, index

    def __len__(self):
        return len(self.text_embeddings)

    @property
    def text_dim(self):
        return self._text_dim

    @property
    def img_dim(self):
        return self._img_dim