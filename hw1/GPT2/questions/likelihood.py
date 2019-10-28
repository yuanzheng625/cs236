import torch
import torch.nn as nn
import torch.nn.functional as F

def log_likelihood(model, text):
    """
    Compute the log-likelihoods for a string `text`
    :param model: The GPT-2 model
    :param texts: A tensor of shape (1, T), where T is the length of the text
    :return: The log-likelihood. It should be a Python scalar. 
        NOTE: for simplicity, you can ignore the likelihood of the first token in `text`.
    """

    with torch.no_grad():
        ## TODO:
        ##  1) Compute the logits from `model`;
        ##  2) Return the log-likelihood of the `text` string. It should be a Python scalar.
        ##      NOTE: for simplicity, you can ignore the likelihood of the first token in `text`
        
        logits, _ = model(text)
        text_from_second_token = text[0, 1:]
        logits = logits[0, :-1, :]
        log_softmax = F.log_softmax(logits, dim=1)
        loglikelihood_from_second_token = log_softmax.gather(1, text_from_second_token.view(-1, 1))
        ll = torch.sum(loglikelihood_from_second_token)
        
        return ll
        

        