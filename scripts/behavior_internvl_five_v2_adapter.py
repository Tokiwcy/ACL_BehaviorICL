"""Independent InternVL2.5 native-image adapter; CPU-safe until explicit load.

No author model code is imported by module import. Runtime execution requires
the separately locked source/weights, isolated runtime, exact Pod, new backup
and science guards. Does not patch the author code or substitute transcripts.
"""
from __future__ import annotations

import copy
import time

SIZE = 448
MAX_TILES = 2  # Native dynamic grid, plus one thumbnail iff more than one tile.
TOKENS_PER_TILE = 256
LAYERS, WIDTH = 32, 4096
NOUNS = {'dtd': 'texture category', 'aircraft': 'aircraft variant', 'cub': 'bird species', 'pets': 'pet breed', 'dogs': 'dog breed'}


def tile_grid(width, height):
    if width <= 0 or height <= 0:
        raise ValueError('Invalid image geometry')
    # Native README algorithm, max_num fixed BEFORE results. Stable integer
    # ordering explicitly frozen for same-area ties rather than set iteration.
    ratios = sorted({(i, j) for n in range(1, MAX_TILES + 1)
                     for i in range(1, n + 1) for j in range(1, n + 1)
                     if 1 <= i * j <= MAX_TILES}, key=lambda x: (x[0] * x[1], x[0], x[1]))
    best, difference = (1, 1), float('inf')
    for ratio in ratios:
        delta = abs(width / height - ratio[0] / ratio[1])
        if delta < difference or (delta == difference and width * height > .5 * SIZE * SIZE * ratio[0] * ratio[1]):
            best, difference = ratio, delta
    return best


def tiles(image):
    image = image.convert('RGB')
    left, top = tile_grid(*image.size)
    resized = image.resize((left * SIZE, top * SIZE))
    values = [resized.crop((i % left * SIZE, i // left * SIZE,
                           (i % left + 1) * SIZE, (i // left + 1) * SIZE))
              for i in range(left * top)]
    if len(values) > 1:
        values.append(image.resize((SIZE, SIZE)))
    if not 1 <= len(values) <= MAX_TILES + 1:
        raise ValueError('Native tile budget exceeded')
    return values


def question(dataset, labels, demo_labels=()):
    if dataset not in NOUNS or not labels or len(set(labels)) != len(labels):
        raise ValueError('Wrong task or legal-label inventory')
    if len(demo_labels) not in (0, 4) or any(v not in labels for v in demo_labels):
        raise ValueError('Exactly zero or four legally labelled demos required')
    noun = NOUNS[dataset]
    if not demo_labels:
        return ('<image>\n' + f'Classify the image into exactly one {noun}. '
                + f"The only valid labels are: {', '.join(labels)}. "
                + 'Return exactly one valid label and no explanation. Label:')
    text = f"Learn the image-to-{noun} task from the demonstrations. The only valid labels are: {', '.join(labels)}.\n"
    for index, label in enumerate(demo_labels, 1):
        text += f'Demonstration {index}:<image>\nLabel: {label}\n'
    return text + 'Query image:<image>\nReturn exactly one valid label and no explanation. Label:'


def render_prompt(model, text, patch_counts):
    if len(patch_counts) not in (1, 5) or text.count('<image>') != len(patch_counts):
        raise ValueError('Native placeholder/ordered-image mismatch')
    if any(not isinstance(v, int) or not 1 <= v <= MAX_TILES + 1 for v in patch_counts):
        raise ValueError('Native patch budget mismatch')
    if int(model.num_image_token) != TOKENS_PER_TILE or model.template != 'internvl2_5':
        raise ValueError('Wrong native model template/visual geometry')
    template = copy.deepcopy(model.conv_template)
    template.messages = []
    template.system_message = model.system_message
    template.append_message(template.roles[0], text)
    template.append_message(template.roles[1], None)
    prompt = template.get_prompt()
    for count in patch_counts:
        prompt = prompt.replace('<image>', '<img>' + '<IMG_CONTEXT>' * (TOKENS_PER_TILE * count) + '</img>', 1)
    if '<image>' in prompt or prompt.count('<IMG_CONTEXT>') != sum(patch_counts) * TOKENS_PER_TILE:
        raise ValueError('Native image replacement count differs; no truncation fallback')
    return prompt, template.sep.strip()


def label_codec(tokenizer, labels, sep):
    eos = int(tokenizer.convert_tokens_to_ids(sep))
    if eos < 0 or eos == tokenizer.unk_token_id or tokenizer.encode(sep, add_special_tokens=False) != [eos]:
        raise ValueError('Native conversation EOS must be one actual token')
    owners = {}
    for label in labels:
        for literal in (label, ' ' + label):
            ids = tuple(int(v) for v in tokenizer.encode(literal, add_special_tokens=False))
            if (not ids or eos in ids or tokenizer.unk_token_id in ids
                    or tokenizer.decode(list(ids), skip_special_tokens=False) not in (label, ' ' + label)):
                raise ValueError('Exact native label token roundtrip failed')
            if ids in owners and owners[ids] != label:
                raise ValueError('Tokenized class labels collide')
            owners[ids] = label
    return owners, eos


class NativePrefix:
    """Measure actual inputs_embeds generation prefix; never assume Qwen offset."""
    def __init__(self, owners, eos):
        from legal_label_decoding import LegalLabelTrie
        self.owners, self.eos = owners, eos
        self.trie = LegalLabelTrie([list(v) for v in owners], eos)
        self.prefix, self.calls = None, 0

    def __call__(self, batch, token_ids):
        if batch != 0:
            raise ValueError('Native generation is batch1 only')
        ids = tuple(int(v) for v in token_ids.tolist())
        if self.prefix is None:
            # Native author generate supplies inputs_embeds only; 4.37.2 creates
            # an empty or one-BOS synthetic token prefix. Must measure it live.
            if len(ids) > 1:
                raise ValueError('Unexpected native embedding-generation initial prefix')
            self.prefix = ids
        if ids[:len(self.prefix)] != self.prefix:
            raise ValueError('Native generation prefix changed')
        self.calls += 1
        return self.trie.allowed(list(ids[len(self.prefix):]))

    def finish(self, output):
        ids = tuple(int(v) for v in output)
        if self.prefix is None or not self.calls or ids[:len(self.prefix)] != self.prefix:
            raise ValueError('Native generation offset was not actually measured')
        tail = ids[len(self.prefix):]
        if len(tail) < 2 or tail[-1] != self.eos or self.eos in tail[:-1] or tail[:-1] not in self.owners:
            raise ValueError('Actual output must be one exact legal label followed by native EOS')
        return self.owners[tail[:-1]], list(tail)


def prepare(model, tokenizer, rows, text):
    from behavior_internvl_policy import live_guard
    live_guard()
    import torch
    from PIL import Image
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    transform = T.Compose([T.Resize((SIZE, SIZE), interpolation=InterpolationMode.BICUBIC),
                           T.ToTensor(), T.Normalize((.485, .456, .406), (.229, .224, .225))])
    pixels, patch_counts = [], []
    for row in rows:
        with Image.open(row['path']) as image:
            pieces = tiles(image)
        pixels.extend(transform(piece) for piece in pieces)
        patch_counts.append(len(pieces))
    pixel_values = torch.stack(pixels).to(device='cuda', dtype=torch.bfloat16)
    prompt, sep = render_prompt(model, text, patch_counts)
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    context_id = int(tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>'))
    if context_id == tokenizer.unk_token_id:
        raise ValueError('Native visual context token absent')
    model.img_context_token_id = context_id
    expected = sum(patch_counts) * TOKENS_PER_TILE
    if int(inputs['input_ids'].eq(context_id).sum()) != expected:
        raise ValueError('Tokenizer visual/context count differs; no fallback')
    token_count = int(inputs['input_ids'].shape[1])
    if token_count + 128 >= int(model.config.llm_config.max_position_embeddings):
        raise ValueError('Native context budget exceeded; do not truncate/retile by outcome')
    return {'pixels': pixel_values, **inputs}, {'prompt': prompt, 'sep': sep,
            'patch_counts': patch_counts, 'image_ids': [row['sample_id'] for row in rows],
            'input_tokens': token_count, 'visual_tokens': expected}


def hidden(model, batch):
    """Native vision -> native language decoder, strictly pre-answer, no labels."""
    from behavior_internvl_policy import live_guard
    live_guard()
    import torch
    if (len(model.language_model.model.layers) != LAYERS
            or int(model.config.llm_config.hidden_size) != WIDTH):
        raise ValueError('Actual native layer/hidden dimensions differ')
    inputs = batch['input_ids']
    visual = model.extract_feature(batch['pixels'])
    selected = inputs.eq(model.img_context_token_id)
    if visual.shape != (len(batch['pixels']), TOKENS_PER_TILE, WIDTH) or int(selected.sum()) != visual.shape[0] * visual.shape[1]:
        raise ValueError('Actual native visual embedding/token mismatch')
    embeds = model.language_model.get_input_embeddings()(inputs).clone()
    embeds[selected] = visual.reshape(-1, WIDTH).to(embeds.dtype)
    captured, hooks = [], []
    def hook(module, args, result):
        value = result[0] if isinstance(result, tuple) else result
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[-1] != WIDTH:
            raise ValueError('Actual decoder hook shape differs')
        captured.append(value[:, -1].detach().clone())
    try:
        for layer in model.language_model.model.layers:
            hooks.append(layer.register_forward_hook(hook))
        with torch.inference_mode():
            result = model.language_model.model(inputs_embeds=embeds, attention_mask=batch['attention_mask'],
                                                 use_cache=False, output_hidden_states=False, return_dict=True)
            if len(captured) != LAYERS:
                raise ValueError('Did not actually capture every native decoder layer exactly once')
            # HF hidden_states[1:] ends in the final norm rather than raw layer32.
            captured[-1] = result.last_hidden_state[:, -1].detach().clone()
            values = torch.stack(captured, 1)
            if values.shape != (1, LAYERS, WIDTH) or not torch.isfinite(values).all():
                raise ValueError('Nonfinite/incorrect pre-answer native response')
        return values[0]
    finally:
        for handle in hooks:
            handle.remove()


def generate(model, tokenizer, batch, metadata, labels):
    from behavior_internvl_policy import live_guard
    live_guard()
    import torch
    owners, eos = label_codec(tokenizer, labels, metadata['sep'])
    prefix = NativePrefix(owners, eos)
    before = time.perf_counter()
    with torch.inference_mode():
        visual = model.extract_feature(batch['pixels'])
        expected = metadata['visual_tokens']
        if visual.shape != (len(batch['pixels']), TOKENS_PER_TILE, WIDTH) or visual.numel() // WIDTH != expected:
            raise ValueError('Actual native generation visual/token mismatch')
        output = model.generate(pixel_values=batch['pixels'], visual_features=visual,
                                input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                do_sample=False, num_beams=1, eos_token_id=eos, pad_token_id=eos,
                                max_new_tokens=max(map(len, owners)) + 1,
                                prefix_allowed_tokens_fn=prefix, return_dict_in_generate=False)
        torch.cuda.synchronize()
        if output.ndim != 2 or output.shape[0] != 1:
            raise ValueError('Wrong actual native generation batch')
        prediction, tokens = prefix.finish(output[0].tolist())
    answer = tokenizer.decode(tokens[:-1], skip_special_tokens=False)
    if answer not in (prediction, ' ' + prediction):
        raise ValueError('Raw output string must roundtrip exactly; no strip-based repair')
    return {'prediction': prediction, 'raw_answer': answer, 'actual_output_token_ids': tokens,
            'output_tokens': len(tokens), 'seconds': time.perf_counter() - before,
            'measured_native_generation_prefix': list(prefix.prefix),
            'actual_prefix_callback_calls': prefix.calls, 'native_eos_token_id': eos,
            'input_tokens': metadata['input_tokens'], 'visual_tokens': metadata['visual_tokens'],
            'patch_counts': metadata['patch_counts']}
