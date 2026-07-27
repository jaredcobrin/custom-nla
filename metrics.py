def compute_fve(reconstructed, original):
    """Fraction of variance explained: 1 - (residual variance / total variance).
    reconstructed, original: [batch, dim] tensors of AR outputs vs true activations."""
    residual = original - reconstructed
    ss_res = (residual.float() ** 2).sum()
    mean = original.float().mean(dim=0, keepdim=True)
    ss_tot = ((original.float() - mean) ** 2).sum()
    return (1 - ss_res / (ss_tot + 1e-8)).item()
