import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# Custom dataset class for DNA sequences
class GeneDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer):
        """
        :param sequences: List of DNA sequences
        :param labels: List of corresponding gene function labels
        :param tokenizer: Function to encode DNA sequences into numerical tensors
        """
        self.sequences = [tokenizer(seq) for seq in sequences]
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# Tokenizer to convert DNA sequence to numerical representation
def dna_tokenizer(sequence):
    base_mapping = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}  # Map each base to an integer
    return torch.tensor([base_mapping.get(base, 4) for base in sequence])


# Model definition
class GeneFunctionPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(GeneFunctionPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # Embedding layer
        _, (hidden, _) = self.lstm(x)  # LSTM
        hidden = hidden[-1]  # Take the last hidden state
        output = self.fc(hidden)  # Fully connected layer for classification
        return output


# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")


# Prediction function
def predict_gene_function(model, sequence, tokenizer, device='cuda'):
    """
    Predicts the function of a given gene sequence using the trained model.
    :param model: Trained PyTorch model
    :param sequence: DNA sequence to be classified
    :param tokenizer: Function to tokenize the sequence
    :param device: Device to run the inference (cuda or cpu)
    :return: Predicted class label
    """
    model.eval()
    sequence_tensor = tokenizer(sequence).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(sequence_tensor)
        predicted_label = torch.argmax(output, dim=1).item()
    return predicted_label


# Example pipeline
if __name__ == "__main__":
    # Sample data
    sequences = ["ATCGGTA", "GGGTTAC", "ATGNNNN", "CGTATCG"]
    labels = [0, 1, 2, 1]  # Example gene function classes

    # Hyperparameters
    vocab_size = 5  # A, T, C, G, N
    embed_dim = 8
    hidden_dim = 16
    num_classes = 3
    batch_size = 2
    num_epochs = 10

    # Prepare dataset and dataloader
    tokenizer = dna_tokenizer
    dataset = GeneDataset(sequences, labels, tokenizer)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    model = GeneFunctionPredictor(vocab_size, embed_dim, hidden_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    # Test prediction
    test_sequence = "GGGTTAC"
    predicted_function = predict_gene_function(model, test_sequence, tokenizer)
    print(f"Predicted Function Class for '{test_sequence}': {predicted_function}")
