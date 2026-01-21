import torch
import torch.nn as nn

class CheXagentSigLIPBinary(nn.Module):
    def __init__(self, vision_encoder):
        super().__init__()

        self.vision_encoder = vision_encoder
        in_dim = vision_encoder.config.hidden_size

        self.view_embedding = nn.Embedding(num_embeddings=3, embedding_dim=8)
        self.sex_embedding  = nn.Embedding(num_embeddings=2, embedding_dim=4)           

        # Binary classifier head
        self.classifier = nn.Sequential(
            nn.Linear(in_dim + 8 + 4, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, inputs, view, sex):
        outputs = self.vision_encoder(pixel_values=inputs)
        embeddings = outputs.pooler_output   
        view_emb = self.view_embedding(view)
        sex_emb = self.sex_embedding(sex)
        combined = torch.cat([embeddings, view_emb, sex_emb], dim=1)
        logits = self.classifier(combined)     
        return logits