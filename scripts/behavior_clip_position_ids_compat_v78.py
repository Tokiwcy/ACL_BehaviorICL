"""Validate two legacy serialized CLIP buffers before strict state loading.

No model is imported or run here. The caller still loads the original SHA-
verified checkpoint with the unchanged strict=True loader. Every learned
tensor retains its original object/bytes; only identical nonpersistent
arange buffers may be omitted from the serialization mapping.
"""
from collections import OrderedDict
import hashlib
import struct

KEYS = ('text_model.embeddings.position_ids', 'vision_model.embeddings.position_ids')


def normalize(model, state, tensor_library):
    if (type(model).__name__ != 'CLIPModel'
            or type(model).__module__ != 'transformers.models.clip.modeling_clip'
            or model.config.vision_config.image_size != 224
            or model.config.vision_config.patch_size != 32
            or model.config.text_config.max_position_embeddings != 77):
        raise ValueError('Exact original CLIP ViT-B/32 interface required')
    expected = model.state_dict()
    if set(state) != set(expected) | set(KEYS) or set(expected) & set(KEYS):
        raise ValueError('Only the exact two legacy nonpersistent buffer keys may differ')
    reports = {}
    for key, count in zip(KEYS, (77, 50)):
        embeddings = model.text_model.embeddings if key == KEYS[0] else model.vision_model.embeddings
        if ('position_ids' not in embeddings._non_persistent_buffers_set
                or 'position_ids' not in embeddings._buffers):
            raise ValueError('Original generated buffer must be nonpersistent')
        saved, generated = state[key], embeddings.position_ids
        reference = tensor_library.arange(count, dtype=tensor_library.int64).expand((1, -1))
        if (saved.dtype != tensor_library.int64 or generated.dtype != tensor_library.int64
                or tuple(saved.shape) != (1, count) or tuple(generated.shape) != (1, count)
                or str(saved.device) != 'cpu' or str(generated.device) != 'cpu'
                or not tensor_library.equal(saved, generated)
                or not tensor_library.equal(saved, reference)):
            raise ValueError('Serialized position IDs must exactly equal the generated integer arange')
        ids = saved.reshape(-1).tolist()
        reports[key] = {'dtype': 'int64', 'shape': [1, count],
                        'numeric_SHA_int64_little_endian': hashlib.sha256(struct.pack('<' + 'q' * count, *ids)).hexdigest(),
                        'exact_generated_buffer_equality': True}
    normalized = OrderedDict((key, value) for key, value in state.items() if key not in KEYS)
    if hasattr(state, '_metadata'):
        normalized._metadata = state._metadata
    if set(normalized) != set(expected) or any(normalized[key] is not state[key] for key in normalized):
        raise ValueError('Learned tensors must remain identical and complete')
    return normalized, {'removed_keys': list(KEYS), 'buffers': reports,
                        'learned_tensor_objects_unchanged': True, 'strict_state_loading_still_required': True,
                        'original_checkpoint_must_remain_SHA_verified': True}


def strict_load(model, state, tensor_library):
    normalized, receipt = normalize(model, state, tensor_library)
    result = model.load_state_dict(normalized, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError('Complete strict state loading required')
    return result, receipt
