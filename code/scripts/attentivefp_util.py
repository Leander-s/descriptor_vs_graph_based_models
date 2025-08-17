from dgl import backend as F


def get_pos_weight(my_dataset):
    num_pos = F.sum(my_dataset.labels, dim=0)
    num_indices = F.tensor(len(my_dataset.labels))
    return (num_indices - num_pos) / num_pos
