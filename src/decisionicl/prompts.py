"""Accepted image-example, CIRCLES and VICL text-demo prompts."""
BASELINE_METHODS = {"random", "rices", "muier", "rices_decisionicl", "muier_decisionicl", "promptself", "promptself_decisionicl", "generic"}

def official_content(question, query, demos, labels, method, attribute="", caption=""):
    prefix = ("Your task is to perform Image Classification. "
              f"You need to choose one of the following options: {list(labels)}")
    content = [
        {"type": "text", "text": prefix},
        {"type": "image", "image": query},
        {"type": "text", "text": f"Question: {question}"},
    ]
    if method in BASELINE_METHODS:
        content.append({"type": "text", "text":
                        f"Here are {len(demos)} in-context examples to help you answer the question:"})
        for image, label in demos:
            content.extend([
                {"type": "image", "image": image},
                {"type": "text", "text": f"Question: {question}\nAnswer: {label}"},
            ])
    else:
        content.extend([
            {"type": "text", "text": "Here are some in-context examples to help you answer the question."},
            {"type": "text", "text": "Examples retrieved based on the original image:"},
            {"type": "image", "image": demos[0][0]},
            {"type": "text", "text": f"Question: {question}\nAnswer: {demos[0][1]}"},
            {"type": "text", "text":
                f'Examples retrieved based on the target image description after changing "{attribute}" (caption: {caption}):'},
            {"type": "image", "image": demos[1][0]},
            {"type": "text", "text": f"Question: {question}\nAnswer: {demos[1][1]}"},
        ])
    content.extend([
        {"type": "text", "text": "Here is the original question again."},
        {"type": "image", "image": query},
        {"type": "text", "text": f"Question: {question}"},
        {"type": "text", "text": "Please provide your response by directly outputting the answer."},
    ])
    return [{"role": "user", "content": content}]


def vicl_one_shot(query_path, summary, demo_label, question, labels):
    prefix = (
        "Your task is to perform Image Classification. "
        f"You need to choose one of the following options: {list(labels)}")
    return [{"role": "user", "content": [
        {"type": "text", "text": prefix},
        {"type": "text", "text":
            f"Question: {question}\nImage 1 summary: {summary}\n"
            f"Answer: {demo_label}\n"
            "Use the example above to answer the same question for Image 2."},
        {"type": "image", "image": str(query_path)},
        {"type": "text", "text":
            f"Question: {question}\n"
            "Please provide your response by directly outputting the answer."},
    ]}]


def vicl_two_shot(query_path, summaries, labels_for_demos, question, labels):
    if len(summaries) != 2 or len(labels_for_demos) != 2:
        raise ValueError('exactly two text demonstrations required')
    prefix = ('Your task is to perform Image Classification. '
              f'You need to choose one of the following options: {list(labels)}')
    examples = ''.join(
        f'Image {i} summary: {summary}\nAnswer: {label}\n'
        for i, (summary, label) in enumerate(zip(summaries, labels_for_demos), 1))
    return [{'role': 'user', 'content': [
        {'type': 'text', 'text': prefix},
        {'type': 'text', 'text': f'Question: {question}\n{examples}'
         'Use the two examples above to answer the same question for Image 3.'},
        {'type': 'image', 'image': str(query_path)},
        {'type': 'text', 'text': f'Question: {question}\n'
         'Please provide your response by directly outputting the answer.'}]}]

QUERY_PROMPT = 'Generate a detailed description of the content depicted in the provided image.'

IOIS_PROMPT = 'Generate a descriptive caption for the provided image and labels, elucidating both the visual content and the underlying purpose or intention depicted. Craft a clear and concise description that seamlessly integrates details from the image and labels, highlighting connections between visual cues and semantic meaning. Your caption should not only describe what is visible in the image but also convey the task-oriented aspect.'
