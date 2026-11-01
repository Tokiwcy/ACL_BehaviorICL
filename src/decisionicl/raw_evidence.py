from .evidence import build_evidence_store, UNCALIBRATED_EVIDENCE_TRANSFORM
POLICY = {"text_template": "This image shows {description}."}

def build_raw_siglip_evidence(*,samples,image_embeddings,bank,encoder):
    if any(s.split not in ("train","validation","test") for s in samples):
        raise ValueError("unexpected evidence split")
    if len({s.split for s in samples})!=1:
        raise ValueError("do not combine evidence splits")
    if "siglip" not in encoder.model_id.lower():
        raise ValueError("this selected policy only supports SigLIP")
    result=build_evidence_store(samples=samples,image_embeddings=image_embeddings,bank=bank,
        encoder=encoder,text_template=POLICY["text_template"],calibration=None)
    validate_raw_policy(result)
    return result


def validate_raw_policy(store):
    store.validate()
    if store.calibration_hash!="identity" or store.transform!=UNCALIBRATED_EVIDENCE_TRANSFORM:
        raise ValueError("selected SigLIP policy forbids train mean debias or a different transform")
    if store.text_template!=POLICY["text_template"]:
        raise ValueError("selected SigLIP policy forbids a different prompt template")
