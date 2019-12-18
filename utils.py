import torch


def predict_by_windows(model, windows, file_type, L2I):
    with open('./data/test_{0}.txt'.format(file_type), mode='w') as file:
        predictions = list()
        for input in windows:
            y = model(input[0])
            _, y = torch.max(y, 1)
            y = L2I[int(y)]
            predictions.append(y)
            file.write("{0}\n".format(y))
    file.close()
    return predictions

def save_model_to_path(model, path="./data/model"):
	torch.save(model.state_dict(), path)


def load_model_from_path(model, path="./data/model"):
	model.load_state_dict(torch.load(path))
