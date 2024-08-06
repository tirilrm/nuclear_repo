import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, threshold=0.8):
        super(CustomLoss, self).__init__()
        self.threshold = threshold
        self.cross_entropy = nn.CrossEntropyLoss()
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def similarity(self, a, b):
        return self.cosine_similarity(a, b)

    def compute_instance_loss(self, pair_embeddings, predictions, triplet_embeddings):
        EMBEDDING_DIM = 768

        pair_heads = pair_embeddings[:, :EMBEDDING_DIM]
        pair_tails = pair_embeddings[:, EMBEDDING_DIM:2 * EMBEDDING_DIM]

        gold_heads = triplet_embeddings[:, :EMBEDDING_DIM]
        gold_tails = triplet_embeddings[:, EMBEDDING_DIM + 1:2 * EMBEDDING_DIM + 1]
        gold_relations = triplet_embeddings[:, EMBEDDING_DIM:EMBEDDING_DIM + 1].long()

        # Compute all pairwise similarities in a batch
        head_similarities = self.similarity(pair_heads.unsqueeze(1), gold_heads.unsqueeze(0))
        tail_similarities = self.similarity(pair_tails.unsqueeze(1), gold_tails.unsqueeze(0))
        avg_similarities = (head_similarities + tail_similarities) / 2

        instance_loss = 0

        for i in range(len(predictions)):
            prediction = predictions[i]  # Tensor of shape [num_classes]
            similarities = avg_similarities[i]

            # Find the best triplet
            best_similarity, best_triplet_idx = torch.max(similarities, dim=0)

            if best_similarity > self.threshold:
                gold_relation = gold_relations[best_triplet_idx].squeeze()
            else:
                gold_relation = torch.tensor(0, dtype=torch.long, device=predictions.device)  # Default relation is '0'

            target = gold_relation.unsqueeze(0)
            prediction = prediction.unsqueeze(0)
            loss = self.cross_entropy(prediction, target)
            instance_loss += loss

        avg_instance_loss = instance_loss / len(predictions)
        return avg_instance_loss

    def forward(self, batch_entity_pairs, batch_predictions, batch_triplets):
        batch_loss = 0
        for entity_pairs, predictions, triplets in zip(batch_entity_pairs, batch_predictions, batch_triplets):
            instance_loss = self.compute_instance_loss(entity_pairs, predictions, triplets)
            batch_loss += instance_loss

        avg_loss = batch_loss / len(batch_predictions)
        return avg_loss
