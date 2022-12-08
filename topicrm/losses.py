import torch

def log_prob(logits, target, mask):
    """
    logits  : shape [batch, seq_len, num_tokens]
    target  : shape [batch, seq_len]
    mask    : shape [batch, seq_len]
    """
    probs = torch.softmax(logits, dim=2)
    target_p = torch.gather(probs, 2, target.unsqueeze(2)).squeeze()
    seq_len_mask = mask.sum(dim=1)
    avg_log_p = ((torch.log(target_p) * mask)/ seq_len_mask.unsqueeze(1)).sum(dim=1) 
    return avg_log_p.mean()

def kl_loss(logits0, logitsf, mask):
    """
    logits0: shape [batch, seq_len, num_tokens]
    logitsf: shape [batch, seq_len, num_tokens]
    """
    probs0 = torch.softmax(logits0, dim=2)
    log_probsf = torch.log_softmax(logitsf, dim=2)
    seq_len_mask = mask.sum(dim=1)
    kl_seq = (probs0 * (torch.log(probs0) - log_probsf)).sum(dim=2)
    avg_kl = (kl_seq * mask).sum(dim=1) / seq_len_mask
    return avg_kl.mean()