import numpy as np
import torch
import clip

def get_templates_for_mnist():
    """
    Get valid prompt templates for zero shot classification on MNIST data

    Returns:
        get_templates (function): Function that returns valid prompt templates for MNIST
    """
    def get_templates(q):
        return [
            f"a photo of the number {q}",
            f"the digit {q}",
            f"an image of a {q}",
            f"{q}",
        ]

    return get_templates

def get_txt_embs(model_id, class_labels, get_templates):
    """
    Use CLIP language encoder to generate text embeddings for zero shot classification

    Parameters: 
        model_id (str): A valid CLIP model ID (e.g. 'clip-rn50')
        class_labels (list): Class labels for zero-shot classification
        get_templates (function): Function that returns valid prompt templates
    Returns:
        txt_embs (torch.Tensor): Torch tensor of size class_labels x emb_dim 
    """
    if model_id == "clip-rn50":
        model, _ = clip.load("RN50", "cuda")
    else: 
        raise NotImplementedError

    all_queries = np.concatenate([get_templates(q) for q in class_labels])
    text = clip.tokenize(all_queries, truncate=True).to("cuda")
    with torch.no_grad():
        text_features = model.encode_text(text).detach().cpu()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    text_to_emb = dict(zip(all_queries, text_features.tolist()))

    txt_embs = []
    for q in class_labels:
        query = get_templates(q)
        txt_emb = torch.tensor(np.stack([text_to_emb[q] for q in query]))
        txt_emb = txt_emb.mean(dim=0, keepdim=True)
        txt_emb /= txt_emb.norm(dim=-1, keepdim=True)
        txt_embs.append(txt_emb)
    txt_embs = torch.cat(txt_embs)
    return txt_embs

def zero_shot_cls(
    img_embs,
    model_id,
    class_labels,
    get_templates,
):
    """
    Perform zero-shot classification.

    Parameters: 
        img_embs (np.Array): Visual embeddings of size num_samples x emb_dim
        model_id (str): A valid CLIP model ID (e.g. 'clip-rn50')
        class_labels (list): Class labels for zero-shot classification
        get_templates (function): Function that returns valid prompt templates
    Returns:
        all_pred_label (np.Array): Predicted label for each sample in img_embs
        pred_score_dist (np.Array): Score distribution (after softmax) across classes for 
                                    each sample in img_embs (size = num_samples x num_classes)
    """
    # Get text embeddings for each class label
    txt_embs = get_txt_embs(model_id, class_labels, get_templates)

    # Compute labels and scores for each image
    sim = (torch.tensor(img_embs).float() @ txt_embs.float().T) / 0.07
    pred_score_dist = sim.softmax(dim=-1)
    all_pred_label = np.array([class_labels[x] for x in torch.argmax(sim, axis=1)])

    return np.array(all_pred_label), pred_score_dist
