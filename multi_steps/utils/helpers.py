
import torch

def save_model(model, optimizer, epoch, filepath):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, filepath)

def load_model(filepath, model, optimizer):
    state = torch.load(filepath)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    return state['epoch']

