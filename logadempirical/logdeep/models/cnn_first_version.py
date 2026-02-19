import torch
import torch.nn as nn

import warnings

warnings.filterwarnings("ignore")


class TextCNN(nn.Module):
    def __init__(self,
                 embedding_dim: int = 300,
                 max_seq_len: int = 100,
                 out_channels: int = 100,
                 n_class: int = 2,
                 dropout: float = 0.5):
        super(TextCNN, self).__init__()

        in_channels = 1
        self.kernel_size_list = [3, 4, 5]
        vocab_size = 30

        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim)
        torch.nn.init.uniform_(self.embedding.weight)
        self.embedding.weight.requires_grad = True

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, embedding_dim)),
            nn.ReLU(),
            nn.MaxPool2d((max_seq_len - kernel_size + 1, 1)),
            nn.Dropout(dropout)
        ) for kernel_size in self.kernel_size_list])

        self.fc = nn.Linear(len(self.kernel_size_list) * out_channels, n_class)

    def forward(self, features, device='cuda'):
        input = features[2]
        # print(input.shape)
        batch_size = input.size(0)
        input = self.embedding(input)
        #input = torch.unsqueeze(input, 1)
        print(input.shape)
        input = [conv(input) for conv in self.convs]
        input = torch.cat(input, dim=1)
        input = input.view(batch_size, -1)
        output = self.fc(input)

        return output, input

if __name__ == '__main__':
    model = TextCNN(300, 100, 8)
    inp = torch.rand(64, 100, 300)
    out, _ = model(inp)
    print(out.shape)
