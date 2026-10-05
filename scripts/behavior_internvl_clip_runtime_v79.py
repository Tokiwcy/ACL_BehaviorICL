"""Narrow compatibility wrapper around the unchanged original load_clip.

This module imports no model/torch. A future independently bound cloud runner
must supply the real pinned modules, checkpoint identity and receipt writer.
It is not a launch entry and does not authorize any GPU execution.
"""
import behavior_clip_position_ids_compat_v78 as compat


def wrapped_load_clip(original, tensor_library, clip_class, write_receipt):
    def load(plan):
        real = clip_class.load_state_dict
        receipts = []
        def guarded(model, state, strict=True, **kwargs):
            if strict is not True or kwargs or receipts:
                raise ValueError('Exactly one original strict CLIP state load required')
            normalized, receipt = compat.normalize(model, state, tensor_library)
            result = real(model, normalized, strict=True)
            if result.missing_keys or result.unexpected_keys:
                raise ValueError('Complete strict state load required')
            receipts.append(receipt)
            return result
        clip_class.load_state_dict = guarded
        try:
            result = original(plan)
        finally:
            clip_class.load_state_dict = real
        if len(receipts) != 1:
            raise ValueError('Original load_clip must perform exactly one state load')
        write_receipt(receipts[0])
        return result
    return load
