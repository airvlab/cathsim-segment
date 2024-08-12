import torch
import torch.nn.functional as F
import torch.utils.data as data


class DummyData(data.Dataset):
    def __init__(self, num_samples, X_shape, seq_len):
        self.num_samples = num_samples
        self.X_shape = X_shape
        self.point_dims = 2
        self.seq_len = seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        X = torch.randn(self.X_shape)

        # Random sequence length between 1 and seq_len (inclusive)
        seq_len = torch.randint(1, self.seq_len + 1, (1,)).item()

        # Create random sequences for c_seq and t_seq
        c_seq = torch.randn(seq_len, self.point_dims)
        t_seq = torch.randn(seq_len, 1)

        # Concatenate c_seq and t_seq along the last dimension
        target_seq = torch.cat([c_seq, t_seq], dim=-1)

        # Create the mask, initially all ones
        target_mask = torch.ones(self.seq_len)

        # Zero out the mask from the end of the current sequence length to seq_len
        target_mask[seq_len:] = 0

        # Pad target_seq to the right shape
        target_seq = F.pad(target_seq, (0, 0, 0, self.seq_len - seq_len))

        return X, target_seq, target_mask
