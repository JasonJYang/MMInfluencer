import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes, word_embedding_dim=300, lstm_hidden_dim=256, 
                 cnn_feature_dim=2048, fusion_dim=512, dropout=0.5):
        super(MultimodalClassifier, self).__init__()
        
        # Text processing components
        self.word_embedding_dim = word_embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        
        # Biography LSTM
        self.bio_lstm = nn.LSTM(
            input_size=word_embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Post text LSTM (shared with biography)
        self.post_text_lstm = self.bio_lstm
        
        # Image processing components
        # Load pre-trained ResNet-50 and remove the classification layer
        resnet = models.resnet50(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        # Freeze the ResNet parameters
        for param in self.image_encoder.parameters():
            param.requires_grad = False
            
        # Multimodal fusion components
        self.bio_projection = nn.Linear(lstm_hidden_dim * 2, fusion_dim)
        self.post_text_projection = nn.Linear(lstm_hidden_dim * 2, fusion_dim)
        self.image_projection = nn.Linear(cnn_feature_dim, fusion_dim)
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes)
        )
        
    def forward(self, bio_embeds, post_text_embeds, post_images):
        """
        Args:
            bio_embeds: Tensor of shape [batch_size, bio_seq_len, embedding_dim]
            post_text_embeds: Tensor of shape [batch_size, num_posts, post_seq_len, embedding_dim]
            post_images: Tensor of shape [batch_size, num_posts, 3, 224, 224]
        """
        batch_size = bio_embeds.size(0)
        num_posts = post_text_embeds.size(1)
        
        # Process biography text
        _, (h_bio, _) = self.bio_lstm(bio_embeds)
        # Combine forward and backward LSTM states
        h_bio = h_bio.view(2, 2, batch_size, self.lstm_hidden_dim)[-1]
        h_bio = h_bio.transpose(0, 1).contiguous().view(batch_size, -1)
        bio_features = self.bio_projection(h_bio)
        
        # Process post texts
        post_text_features_list = []
        for i in range(num_posts):
            _, (h_post, _) = self.post_text_lstm(post_text_embeds[:, i])
            h_post = h_post.view(2, 2, batch_size, self.lstm_hidden_dim)[-1]
            h_post = h_post.transpose(0, 1).contiguous().view(batch_size, -1)
            post_text_features_list.append(h_post)
        
        # Average post text features
        if post_text_features_list:
            post_text_features = torch.stack(post_text_features_list, dim=1)
            post_text_features = torch.mean(post_text_features, dim=1)
        else:
            post_text_features = torch.zeros(batch_size, self.lstm_hidden_dim * 2).to(bio_embeds.device)
        
        post_text_features = self.post_text_projection(post_text_features)
        
        # Process post images
        post_image_features_list = []
        for i in range(num_posts):
            # Reshape to process each image
            batch_images = post_images[:, i]
            image_features = self.image_encoder(batch_images)
            image_features = image_features.view(batch_size, -1)
            post_image_features_list.append(image_features)
        
        # Average image features
        if post_image_features_list:
            post_image_features = torch.stack(post_image_features_list, dim=1)
            post_image_features = torch.mean(post_image_features, dim=1)
        else:
            post_image_features = torch.zeros(batch_size, 2048).to(bio_embeds.device)
            
        post_image_features = self.image_projection(post_image_features)
        
        # Concatenate and classify
        combined_features = torch.cat([bio_features, post_text_features, post_image_features], dim=1)
        logits = self.classifier(combined_features)
        
        return logits