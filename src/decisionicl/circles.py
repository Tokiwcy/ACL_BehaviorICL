"""Accepted CIRCLES attribute/description parsing and generation prompts."""
INSTRUCTION = 'You are an image description expert. You are given an original image and manipulation text. Your goal is to generate a target image description that reflects the changes described based on manipulation intents while retaining as much image content from the original image as possible.\n\n## Guidelines on generating the Original Image Description\n- Ensure the original image description is thorough, capturing all visible objects, attributes, and elements.\n- The original image description should be as accurate as possible, reflecting the content of the image.\n\n## Guidelines on generating the Thoughts\n- In your Thoughts, explain your understanding of the manipulation intents and how you formulated the target image description.\n- Provide insight into how you interpreted the manipulation intent in detail in the manipulation text.\n- Discuss how the manipulation intent influenced which elements of the original image you focused on.\n\n## Guidelines on generating the Reflections\n- In your Reflections, summarize how the manipulation intent influenced your approach to transforming the original image description.\n- Explain how the changes made reflect the specific semantic, Highlight key decisions that were made to preserve the coherence and context of the original image while meeting the manipulation intent.\n- Reflect on the impact these changes have on the overall appearance or narrative of the image.\n- Ensure that your reflections provide a concise yet insightful summary of the considerations and strategies applied in crafting the target description, offering a logical connection between the original and final content.\n\n## Guidelines on generating Target Image Description\n- The target image description you generate should be complete and can cover various semantic aspects.\n- The target image description only contains the target image content and needs to be as simple as possible. Minimize aesthetic descriptions as much as possible.\n\n## On the input format <Input>\n- Input consist of two parts: The original image and the manipulation text.\n{\n  "Original Image": <image_url>,\n  "Manipulation Text": <manipulation_text>\n}\n\n## Guidelines on determining the response <Response>\n- Responses include the Original Image Context, Target Image Description, and Thoughts.\n{\n  "Original Image Description": <original_image_description>,\n  "Thoughts": <thoughts>,\n  "Reflections": <reflections>,\n  "Target Image Description": <target_image_description>\n}'

def parse_attribute(text):
    for raw in text.split("### Attributes")[-1].strip().splitlines():
        value = raw.strip().lstrip("-*• ").strip()
        while value and value[0].isdigit():
            value = value[1:].lstrip(".):- ")
        if value:
            return value[:240]
    return ""

def parse_caption(text):
    value = text.strip()
    if "Target Image Description" in value:
        value = value.split("Target Image Description")[-1]
    return value.strip('\":} \n\r\t')[:1200]

def describe(image_path, question, backend):
    first = [{"role": "user", "content": [
        {"type":"text","text":"Identify the key attributes of the following image that are most relevant to answering the question."},
        {"type":"image","image":str(image_path)},
        {"type":"text","text":f"Question: {question}"},
        {"type":"text","text":"Please list the top 1 key attributes as short phrases in a section named '### Attributes', one per line, ordered from most to least important."}]}]
    attribute_raw=backend.generate([first])[0]['text']
    attribute=parse_attribute(attribute_raw)
    if not attribute: raise ValueError("empty CIRCLES attribute")
    second=[{"role":"user","content":[{"type":"text","text":INSTRUCTION},
        {"type":"image","image":str(image_path)},
        {"type":"text","text":f"Manipulation Text: Change the attribute '{attribute}' to a different plausible value. Ensure the modified caption is concise and contains no more than 77 tokens."}]}]
    caption_raw=backend.generate([second])[0]['text']
    caption=parse_caption(caption_raw)
    if not caption: raise ValueError("empty CIRCLES caption")
    return dict(attribute=attribute,modified_caption=caption,attribute_raw=attribute_raw,caption_raw=caption_raw)
