from uitls.dataset import create_dataset


def train():
    """
    read the data
    for every epoch, iterate the dataset and do the backpropagation
    """
    dataset, dataloader = create_dataset()
