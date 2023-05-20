import torch
from torch.utils.data import DataLoader


class Test(torch.utils.data.IterableDataset):

    def __iter__(self):
        return iter(range(60))


ds = Test()
dl = DataLoader(ds, batch_size=10, shuffle=True)
for batch in dl:
    print(batch)
