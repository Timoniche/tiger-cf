import torch


class NDCGMetric:
    def __init__(self, k, allowed_item_mask=None):
        self._k = k
        self._allowed_item_mask = allowed_item_mask  # torch.BoolTensor of shape (num_items,) or None

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix][:, :self._k].long()  # (batch_size, top_k_indices)
        labels = inputs[f'{labels_prefix}.ids'].long()  # (batch_size)

        assert labels.shape[0] == predictions.shape[0]

        if self._allowed_item_mask is not None:
            mask = self._allowed_item_mask.to(labels.device)[labels]
            if mask.sum().item() == 0:
                return []
            predictions = predictions[mask]
            labels = labels[mask]

        hits = torch.eq(predictions, labels[..., None]).float()  # (batch_size, top_k_indices)
        discount_factor = 1. / torch.log2(torch.arange(1, self._k + 1).float() + 1.).to(hits.device)  # (k)
        dcg = torch.einsum('bk,k->b', hits, discount_factor)  # (batch_size)

        return dcg.cpu().tolist()


class NDCGSemanticMetric:
    def __init__(self, k, codebook_size, num_codebooks, allowed_item_mask=None):
        self._k = k
        self._codebook_size = codebook_size
        self._num_codebooks = num_codebooks
        self._allowed_item_mask = allowed_item_mask  # torch.BoolTensor of shape (num_items,) or None

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix].long()

        batch_size, _, sid_length = predictions.shape

        # Optionally filter by original item ids
        if self._allowed_item_mask is not None:
            raw_labels = inputs[f'{labels_prefix}.ids'].long()
            keep_mask = self._allowed_item_mask.to(raw_labels.device)[raw_labels]
            if keep_mask.sum().item() == 0:
                return []
            # Reshape semantic labels to (batch_size, sid_length) before masking by batch
            labels2d = inputs[f'semantic_{labels_prefix}.ids'].long().reshape(batch_size, sid_length)
            predictions = predictions[keep_mask]
            labels = labels2d[keep_mask]
            batch_size = keep_mask.sum().item()
        else:
            labels = inputs[f'semantic_{labels_prefix}.ids'].long().reshape(batch_size, sid_length)

        labels = labels[:, None, :]
        offsetted_labels = labels + self._codebook_size * torch.arange(self._num_codebooks, device=labels.device)[None, None, :]

        hits = (torch.eq(predictions[:, :self._k, :], offsetted_labels).sum(dim=-1) == sid_length).float()  # (batch_size, top_k_indices)

        discount_factor = 1 / torch.log2(torch.arange(1, self._k + 1, 1).float() + 1.).to(hits.device)  # (k)
        dcg = torch.einsum('bk,k->b', hits, discount_factor)  # (batch_size)

        return dcg.cpu().tolist()


class RecallMetric:
    def __init__(self, k, allowed_item_mask=None):
        self._k = k
        self._allowed_item_mask = allowed_item_mask  # torch.BoolTensor of shape (num_items,) or None

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix][:, :self._k].long()  # (batch_size, top_k_indices)
        labels = inputs[f'{labels_prefix}.ids'].long()  # (batch_size)

        assert labels.shape[0] == predictions.shape[0]

        if self._allowed_item_mask is not None:
            mask = self._allowed_item_mask.to(labels.device)[labels]
            if mask.sum().item() == 0:
                return []
            predictions = predictions[mask]
            labels = labels[mask]

        hits = torch.eq(predictions, labels[..., None]).float()  # (batch_size, top_k_indices)
        recall = hits.sum(dim=-1)  # (batch_size)

        return recall.cpu().tolist()


class RecallSemanticMetric:
    def __init__(self, k, codebook_size, num_codebooks, allowed_item_mask=None):
        self._k = k
        self._codebook_size = codebook_size
        self._num_codebooks = num_codebooks
        self._allowed_item_mask = allowed_item_mask  # torch.BoolTensor of shape (num_items,) or None

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix].long()

        batch_size, _, sid_length = predictions.shape

        # Optionally filter by original item ids
        if self._allowed_item_mask is not None:
            raw_labels = inputs[f'{labels_prefix}.ids'].long()
            keep_mask = self._allowed_item_mask.to(raw_labels.device)[raw_labels]
            if keep_mask.sum().item() == 0:
                return []
            # Reshape semantic labels to (batch_size, sid_length) before masking by batch
            labels2d = inputs[f'semantic_{labels_prefix}.ids'].long().reshape(batch_size, sid_length)
            predictions = predictions[keep_mask]
            labels = labels2d[keep_mask]
            batch_size = keep_mask.sum().item()
        else:
            labels = inputs[f'semantic_{labels_prefix}.ids'].long().reshape(batch_size, sid_length)

        labels = labels[:, None, :]
        offsetted_labels = labels + self._codebook_size * torch.arange(self._num_codebooks, device=labels.device)[None, None, :]

        hits = (torch.eq(predictions[:, :self._k, :], offsetted_labels).sum(dim=-1) == sid_length).float()  # (batch_size, top_k_indices)
        recall = hits.sum(dim=-1)  # (batch_size)

        return recall.cpu().tolist()
