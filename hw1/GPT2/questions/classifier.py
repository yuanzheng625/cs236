import torch
from questions.likelihood import log_likelihood

def classification(model, text):
    """
    Classify whether the string `text` is randomly generated or not.
    :param model: The GPT-2 model
    :param texts: A tensor of shape (1, T), where T is the length of the text
    :return: True if `text` is a random string. Otherwise return False
    """

    with torch.no_grad():
        ## TODO: Return True if `text` is a random string. Or else return False.
        ll = log_likelihood(model, text)
        if ll < - 4500:
            return True
        else:
            return False