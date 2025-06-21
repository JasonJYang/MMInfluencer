import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import warnings
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
warnings.filterwarnings('ignore')


class MultimodalDataset(Dataset):
    def __init__(self, label_df, user2biography_text_dict, user2post_text_dict, 
                 user2post_image_dir_dict, word_embedding_processor, target_entity_num=None,
                 max_posts=5, max_bio_length=100, max_post_length=100,
                 transform=None):
        self.label_df = label_df
        self.user2biography_text_dict = user2biography_text_dict
        self.user2post_text_dict = user2post_text_dict
        self.user2post_image_dir_dict = user2post_image_dir_dict
        self.word_embedding_processor = word_embedding_processor
        self.max_posts = max_posts
        self.max_bio_length = max_bio_length
        self.max_post_length = max_post_length
        self.target_entity_num = target_entity_num

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # standard normalization for ImageNet pre-trained models
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
        
    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self, idx):
        row = self.label_df.iloc[idx]
        user = row['user']
        user_id = row['user_id']
        target_entity_id = row['target_entity_id']

        # Process biography text
        bio_text = self.user2biography_text_dict.get(user, "")
        bio_embeddings = self.word_embedding_processor.text_to_embeddings(
            bio_text, max_length=self.max_bio_length)
        bio_embeddings = torch.FloatTensor(bio_embeddings)
        
        # Process post texts and images
        post_text_list = self.user2post_text_dict.get(user, [])
        post_image_dir_list = self.user2post_image_dir_dict.get(user, [])
        
        # Ensure we have same number of posts and images
        post_count = min(len(post_text_list), len(post_image_dir_list), self.max_posts)
        
        post_text_embeddings = []
        post_images = []
        
        for i in range(post_count):
            # Process post text
            post_text = post_text_list[i]
            post_embedding = self.word_embedding_processor.text_to_embeddings(
                post_text, max_length=self.max_post_length)
            post_text_embeddings.append(torch.FloatTensor(post_embedding))
            
            # Process post image
            image_path = post_image_dir_list[i]
            try:
                image = Image.open(image_path).convert('RGB')
                image_tensor = self.transform(image)
                post_images.append(image_tensor)
            except Exception as e:
                # If there's an issue loading the image, use a blank one
                blank_image = torch.zeros(3, 224, 224)
                post_images.append(blank_image)
        
        # Pad with empty posts if needed
        while len(post_text_embeddings) < self.max_posts:
            empty_text = torch.zeros(self.max_post_length, bio_embeddings.shape[1])
            post_text_embeddings.append(empty_text)
            
            empty_image = torch.zeros(3, 224, 224)
            post_images.append(empty_image)
        
        # Stack all post text embeddings and images
        post_text_embeddings = torch.stack(post_text_embeddings)
        post_images = torch.stack(post_images)

        return bio_embeddings, post_text_embeddings, post_images, user_id, target_entity_id


class WordEmbeddingProcessor:
    def __init__(self, embedding_dim=300, min_count=5, window=5, workers=4):
        self.embedding_dim = embedding_dim
        self.min_count = min_count
        self.window = window
        self.workers = workers
        self.word2vec_model = None
        self.vocab = None
    
    def train(self, text_corpus):
        """Train Word2Vec model on the provided text corpus
        
        Args:
            text_corpus: List of tokenized texts
        """
        # Convert text to tokens if not already tokenized
        processed_corpus = [
            simple_preprocess(doc) if isinstance(doc, str) else doc 
            for doc in text_corpus
        ]
        
        # Train Word2Vec model
        self.word2vec_model = Word2Vec(
            sentences=processed_corpus,
            vector_size=self.embedding_dim,
            min_count=self.min_count,
            window=self.window,
            workers=self.workers
        )
        
        self.vocab = self.word2vec_model.wv.key_to_index
        
    def text_to_embeddings(self, text, max_length=100):
        """Convert text to sequence of word embeddings
        
        Args:
            text: String or list of tokens
            max_length: Maximum sequence length
        
        Returns:
            numpy array of shape (max_length, embedding_dim)
        """
        if self.word2vec_model is None:
            raise ValueError("Word2Vec model not trained yet")
        
        # Handle different input types and errors
        if not text or not isinstance(text, (str, list)):
            # Return zeros if text is empty or invalid type
            return np.zeros((max_length, self.embedding_dim))
        
        # Tokenize if input is string
        if isinstance(text, str):
            tokens = simple_preprocess(text)
        else:
            tokens = text
            
        # Truncate or pad sequence
        tokens = tokens[:max_length]
        padding_length = max_length - len(tokens)
        
        # Get embeddings for each token
        embeddings = []
        for token in tokens:
            if token in self.vocab:
                embeddings.append(self.word2vec_model.wv[token])
            else:
                # Use random embedding for OOV words
                embeddings.append(np.random.randn(self.embedding_dim))
        
        # Pad sequence
        for _ in range(padding_length):
            embeddings.append(np.zeros(self.embedding_dim))
            
        return np.array(embeddings)
    

class MultimodalDataLoader():
    def __init__(self, logger,
                        data_dir, 
                        use_multimodal=True,
                        label_relationship='suitable_category',
                        target_entity_name='product_category',
                        batch_size=32, 
                        seed=42, 
                        shuffle=True, 
                        validation_split=0.1, 
                        test_split=0.2, 
                        num_workers=1,
                        embedding_dim=300,
                        max_posts=5,
                        max_bio_length=100,
                        max_post_length=100):
            self.logger = logger
            self.data_dir = Path(data_dir)
            self.save_dir = self.data_dir.parent.joinpath('processed')
            self.label_relationship = label_relationship
            self.target_entity_name = target_entity_name
            self.use_multimodal = use_multimodal
            self.logger.info('Using multimodal data: {} with {} as label relationship and {} as target entity name'.format(
                self.use_multimodal, self.label_relationship, self.target_entity_name))
            self.batch_size = batch_size
            self.seed = seed
            self.shuffle = shuffle
            self.validation_split = validation_split
            self.test_split = test_split
            self.num_workers = num_workers
            self.embedding_dim = embedding_dim
            self.max_posts = max_posts
            self.max_bio_length = max_bio_length
            self.max_post_length = max_post_length

            random.seed(self.seed)

            self.label_df, self.user2biography_text_dict, self.user2post_text_dict, self.user2post_image_dir_dict = self._process_data()
            
            # Prepare text embedding model
            self.logger.info('Training Word2Vec model...')
            self.word_embedding_processor = self._train_word_embeddings()            
            self.user2id = {user: idx for idx, user in enumerate(self.label_df['user'].unique())}
            self.target_entity2id = {product: idx for idx, product in enumerate(self.label_df['target_entity'].unique())}
            self.id2user = {v: k for k, v in self.user2id.items()}
            self.id2target_entity = {v: k for k, v in self.target_entity2id.items()}
            
            self.user_num = len(self.user2id)
            self.target_entity_num = len(self.target_entity2id)
            self.label_df['user_id'] = self.label_df['user'].map(self.user2id)
            self.label_df['target_entity_id'] = self.label_df['target_entity'].map(self.target_entity2id)
            self.user_id_list = self.label_df['user_id'].unique().tolist()
            self.target_entity_id_list = self.label_df['target_entity_id'].unique().tolist()

            self.logger.info('Removing duplicates from label dataframe...')
            self.label_df = self.label_df.drop_duplicates(subset=['user_id'])
            self.logger.info('Total {} unique users and {} unique products.'.format(
                self.label_df['user_id'].nunique(), self.label_df['target_entity_id'].nunique()))
            
    def _train_word_embeddings(self):        
        # Collect all text data for training Word2Vec
        all_texts = []
        
        # Add biography texts
        for text in self.user2biography_text_dict.values():
            if text and isinstance(text, str):
                all_texts.append(text)
        
        # Add post texts
        for post_list in self.user2post_text_dict.values():
            for post in post_list:
                if post and isinstance(post, str):
                    all_texts.append(post)
        
        # Initialize and train the embedding processor
        word_embedding_processor = WordEmbeddingProcessor(embedding_dim=self.embedding_dim)
        word_embedding_processor.train(all_texts)
        
        return word_embedding_processor

    def get_user_num(self):
        return self.user_num

    def get_entity_num(self):
        return self.target_entity_num
    
    def _process_data(self):
        self.logger.info('Loading processed data...')
        label_save_dir = self.save_dir.joinpath('labels-{}.csv'.format(self.label_relationship))
        label_df = pd.read_csv(label_save_dir)
        self.logger.info('{} unique influencers and {} unique products loaded.'.format(
            label_df['user'].nunique(), label_df['target_entity'].nunique()))
        
        self.logger.info('Loading biography data...')
        biography_df = pd.read_csv(self.data_dir.parent.joinpath('raw', 'biography.csv'))
        biography_df = biography_df[biography_df['username'].isin(label_df['user'])]
        self.logger.info('Among {} influencers, {} have biography data.'.format(
            label_df['user'].nunique(), biography_df['username'].nunique()))
        user2biography_text_dict = dict(zip(biography_df['username'], biography_df['biography']))
        label_df = label_df[label_df['user'].isin(biography_df['username'])]
        self.logger.info('After filtering, {} unique influencers and {} unique products remain.'.format(
            label_df['user'].nunique(), label_df['target_entity'].nunique()))
        
        self.logger.info('Loading posts data...')
        post_dir = self.data_dir.parent.parent.joinpath('20250613_RawFiles')
        # get the folder name under the post_dir
        influencer_with_post_dir_list = [f for f in post_dir.iterdir() if f.is_dir()]
        influencer_with_post_name_list = [f.name for f in influencer_with_post_dir_list]
        self.logger.info('Found {} influencers with posts.'.format(len(influencer_with_post_name_list)))
        label_df = label_df[label_df['user'].isin(influencer_with_post_name_list)]
        self.logger.info('After filtering, {} unique influencers and {} unique products remain.'.format(
            label_df['user'].nunique(), label_df['target_entity'].nunique()))
        user2post_text_dict, user2post_image_dir_dict = {}, {}
        for influencer_dir in influencer_with_post_dir_list:
            influencer_name = influencer_dir.name
            if influencer_name in label_df['user'].values:
                # get all files ending with .txt in the influencer_dir
                post_files = list(influencer_dir.glob('*.txt'))
                post_files.sort(key=lambda x: x.name)  # sort files by name
                post_text_list = []
                for post_file in post_files:
                    with open(post_file, 'r', encoding='utf-8') as f:
                        post_text = f.read().strip()
                        if post_text:
                            post_text_list.append(post_text)
                user2post_text_dict[influencer_name] = post_text_list

                # get all files ending with .jpg in the influencer_dir
                post_image_files = list(influencer_dir.glob('*.jpg'))
                post_image_files.sort(key=lambda x: x.name)  # sort files by name
                post_image_dir_list = [str(post_image_file) for post_image_file in post_image_files]
                user2post_image_dir_dict[influencer_name] = post_image_dir_list
        self.logger.info('Loaded posts data for {} influencers.'.format(len(user2post_text_dict)))

        return label_df, user2biography_text_dict, user2post_text_dict, user2post_image_dir_dict
    
    def _negative_sampling(self, label_df, user_id_list, target_entity_id_list, id2user, id2target_entity):
        self.logger.info('Negative sampling for each user...')
        neg_label_df_dict = {'user_id': [], 'target_entity_id': [], 'label': []}
        for user in list(set(label_df['user_id'])):
            user_label_df = label_df[label_df['user_id'] == user]
            neg_product_category_list = list(set(target_entity_id_list) - set(user_label_df['target_entity_id']))
            neg_product_category_list = random.sample(neg_product_category_list, len(user_label_df))
            neg_label_df_dict['user_id'] += [user] * len(neg_product_category_list)
            neg_label_df_dict['target_entity_id'] += neg_product_category_list
            neg_label_df_dict['label'] += [0] * len(neg_product_category_list)
        neg_label_df = pd.DataFrame(neg_label_df_dict)
        neg_label_df['user'] = neg_label_df['user_id'].map(id2user)
        neg_label_df['target_entity'] = neg_label_df['target_entity_id'].map(id2target_entity)
        self.logger.info('{} negative samples generated.'.format(neg_label_df.shape))
        return neg_label_df
    
    def get_dataloader(self):
        self.logger.info('Creating dataloader...')
        # label_df = pd.concat([self.label_df, self.neg_label_df], ignore_index=True)
        label_df = self.label_df.copy()
        label_df = label_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        train_num = int(len(label_df) * (1 - self.test_split - self.validation_split))
        test_num = int(len(label_df) * self.test_split)
        train_df = label_df[:train_num]
        test_df = label_df[train_num:train_num+test_num]
        valid_df = label_df[train_num+test_num:]
        self.logger.info('Train: {}, Valid: {}, Test: {}'.format(train_df.shape, valid_df.shape, test_df.shape))

        self.logger.info('Creating datasets...')
        if self.use_multimodal:
            train_dataset = MultimodalDataset(
                train_df, self.user2biography_text_dict, self.user2post_text_dict, self.user2post_image_dir_dict,
                self.word_embedding_processor, max_posts=self.max_posts, target_entity_num=self.target_entity_num,
                max_bio_length=self.max_bio_length, max_post_length=self.max_post_length)
            
            valid_dataset = MultimodalDataset(
                valid_df, self.user2biography_text_dict, self.user2post_text_dict, self.user2post_image_dir_dict,
                self.word_embedding_processor, max_posts=self.max_posts, target_entity_num=self.target_entity_num,
                max_bio_length=self.max_bio_length, max_post_length=self.max_post_length)
            
            test_dataset = MultimodalDataset(
                test_df, self.user2biography_text_dict, self.user2post_text_dict, self.user2post_image_dir_dict,
                self.word_embedding_processor, max_posts=self.max_posts, target_entity_num=self.target_entity_num,
                max_bio_length=self.max_bio_length, max_post_length=self.max_post_length)

        self.logger.info('Creating dataloaders...')
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader, valid_loader, test_loader
    
    def get_dataloader_kfold(self, K):
        self.logger.info('Creating dataloader for {}-fold...'.format(K))
        # label_df = pd.concat([self.label_df, self.neg_label_df], ignore_index=True)
        label_df = self.label_df.copy()
        label_df = label_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        label_index_list = list(range(len(label_df)))
        np.random.seed(self.seed)
        np.random.shuffle(label_index_list)
        fold_size = len(label_index_list) // K
        fold_index_list = [label_index_list[i*fold_size: (i+1)*fold_size] for i in range(K)]
        if len(fold_index_list) % K != 0:
            fold_index_list[-1] += label_index_list[K*fold_size:]
    
        kfold_index_dict = {}
        for fold_idx in range(K):
            test_index_list = fold_index_list[fold_idx]
            train_index_list = []
            for i in range(K):
                if i != fold_idx:
                    train_index_list += fold_index_list[i]
            # 10% of training data is used for validation
            np.random.shuffle(train_index_list)
            valid_index_list = train_index_list[: int(len(train_index_list)*0.1)]
            train_index_list = train_index_list[int(len(train_index_list)*0.1): ]
            kfold_index_dict[fold_idx] = (train_index_list, valid_index_list, test_index_list)

        dataloader_list = []
        for fold_idx in range(K):
            train_index_list, valid_index_list, test_index_list = kfold_index_dict[fold_idx]
            train_df = label_df.iloc[train_index_list]
            valid_df = label_df.iloc[valid_index_list]
            test_df = label_df.iloc[test_index_list]
            self.logger.info('Fold {}: Train: {}, Valid: {}, Test: {}'.format(fold_idx, train_df.shape, valid_df.shape, test_df.shape))

            self.logger.info('Creating datasets...')
            if self.use_multimodal:
                train_dataset = MultimodalDataset(
                    train_df, self.user2biography_text_dict, self.user2post_text_dict, self.user2post_image_dir_dict,
                    self.word_embedding_processor, max_posts=self.max_posts, target_entity_num=self.target_entity_num,
                    max_bio_length=self.max_bio_length, max_post_length=self.max_post_length)
                
                valid_dataset = MultimodalDataset(
                    valid_df, self.user2biography_text_dict, self.user2post_text_dict, self.user2post_image_dir_dict,
                    self.word_embedding_processor, max_posts=self.max_posts, target_entity_num=self.target_entity_num,
                    max_bio_length=self.max_bio_length, max_post_length=self.max_post_length)
                
                test_dataset = MultimodalDataset(
                    test_df, self.user2biography_text_dict, self.user2post_text_dict, self.user2post_image_dir_dict,
                    self.word_embedding_processor, max_posts=self.max_posts, target_entity_num=self.target_entity_num,
                    max_bio_length=self.max_bio_length, max_post_length=self.max_post_length)

            self.logger.info('Creating dataloaders...')
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
            valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            dataloader_list.append((train_loader, valid_loader, test_loader))

        return dataloader_list
