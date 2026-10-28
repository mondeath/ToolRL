"""Sampling/log-prob probability-space contracts for on-policy GRPO."""


def validate_sampling_logprob_alignment(sampling, logprob_space):
    """Reject behavior transforms that are absent from stored old log-probs.

    V2 stores raw model-softmax log-probabilities.  Sampling is therefore
    allowed only when temperature, nucleus, and top-k transforms are neutral.
    """
    required = {'do_sample', 'temperature', 'top_p', 'top_k'}
    missing = required - set(sampling)
    if missing:
        raise ValueError('Sampling/log-prob alignment missing fields: ' + ', '.join(sorted(missing)))
    if logprob_space != 'raw_model_softmax':
        raise ValueError('Unsupported log-prob space: ' + str(logprob_space))
    expected = dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0)
    actual = {key: sampling[key] for key in expected}
    if actual != expected:
        raise ValueError('Sampling/log-prob space mismatch: raw_model_softmax requires ' + str(expected))
    return dict(status='aligned', behavior_policy='raw_model_softmax', logprob_space=logprob_space,
                sampling=actual)
