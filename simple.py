import torch
import torch.nn as nn

# Positional Encoding for sequence information
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# Minimal Knowledge Tracing model
class SimpleKT(nn.Module):
    def __init__(self, num_concepts, num_questions, d_model=64, num_layers=1):
        super().__init__()
        self.concept_embedding = nn.Embedding(num_concepts, d_model)
        self.question_embedding = nn.Embedding(num_questions, d_model)
        self.score_embedding = nn.Linear(1, d_model)  # score is a float in [0, 1]
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, concept_ids, question_ids, scores):
        # Inputs: (batch, seq_len)
        c_emb = self.concept_embedding(concept_ids)
        q_emb = self.question_embedding(question_ids)
        s_emb = self.score_embedding(scores.unsqueeze(-1))  # (batch, seq_len, 1) → (batch, seq_len, d_model)

        x = c_emb + q_emb + s_emb
        x = self.pos_encoder(x)
        x = self.transformer(x)
        out = self.predictor(x)  # (batch, seq_len, 1)
        return out.squeeze(-1)

# Example usage
if __name__ == "__main__":
    model = SimpleKT(num_concepts=100, num_questions=100)
    batch_size, seq_len = 1, 1
    concept_ids = torch.randint(0, 100, (batch_size, seq_len))
    question_ids = torch.randint(0, 100, (batch_size, seq_len))
    scores = torch.rand(batch_size, seq_len)  # values between 0 and 1
    print(scores)
    output = model(concept_ids, question_ids, scores)
    print("Predicted Scores:\n", output)
