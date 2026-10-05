"""Small Qwen3 adapter for official TAM; no model import until cloud worker.

Official functions are compiled from unchanged pinned source. One documented
first-answer index repair selects the generated token at L, not the last input
token at L-1. Prompt maps are still processed in original order for ECI.
"""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

EXPECTED_SHA='18bc233edf416de0846b09f441a7bcf5496916f500da49014ceb3ec8c81f7067'
def load(path,torch,np,minimize_scalar):
    source=Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest()!=EXPECTED_SHA:raise ValueError('Official pinned TAM bytes differ')
    tree=ast.parse(source);changes=[]
    # Repair only the first-generated-token class index. The author's raw
    # round-0 loop otherwise chooses the final input newline token.
    class Repair(ast.NodeTransformer):
        def visit_If(self,node):
            if ast.unparse(node.test)=='prompt_token_idx == len(prompt)':
                targets=[x for x in node.body if isinstance(x,ast.Assign) and ast.unparse(x.targets[0])=='cls_id']
                if len(targets)!=1 or ast.unparse(targets[0].value)!='tokens[this_token_idx]':raise ValueError('Unexpected TAM first-target implementation')
                targets[0].value=ast.parse('tokens[this_token_idx + 1]',mode='eval').body
                changes.append('first generated class index L instead of L-1')
            return self.generic_visit(node)
    tree=Repair().visit(tree);ast.fix_missing_locations(tree)
    if changes!=['first generated class index L instead of L-1']:raise ValueError('Exact one-line TAM target repair required')
    # eval_only requires only RGB-to-BGR, which does not affect numerical maps.
    # Do not install LaTeX/OpenCV/PyMuPDF just for unused text-rendering paths.
    cv2=SimpleNamespace(COLOR_RGB2BGR=1,cvtColor=lambda value,code:value[...,::-1])
    namespace={'torch':torch,'np':np,'minimize_scalar':minimize_scalar,'cv2':cv2,'Path':Path}
    functions=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[])
    exec(compile(functions,'pinned_official_TAM_eval_adapter','exec'),namespace)
    return namespace,{'official_SHA':EXPECTED_SHA,'adapter_change':changes,
        'author_ECI_and_rank_filter_AST_unchanged':True,'author_source_bytes_changed':False,
        'text_visualization_not_called':True}


def maps(namespace,tokens,shape,logits,special,vision,processor,rounds):
    import numpy as np
    fn=namespace['TAM'];idx=namespace['id2idx']
    start=idx(tokens,special['prompt_id'][0],True)+1
    end=idx(tokens,special['prompt_id'][1])
    answer=idx(tokens,special['answer_id'][0],True)+1
    if not 0<=start<end<answer or answer!=logits[0].shape[1]:
        raise ValueError('Exact actual input/prompt/answer token boundaries required')
    prompt_ids=tokens[start:end]
    decoded=processor.batch_decode([prompt_ids],skip_special_tokens=False,clean_up_tokenization_spaces=False)[0]
    pieces=processor.tokenizer.tokenize(decoded)
    if processor.tokenizer.encode(decoded,add_special_tokens=False)!=prompt_ids or len(pieces)!=len(prompt_ids):
        raise ValueError('Official TAM decoded/retokenized prompt changes token positions')
    raw_records=[];capture=[];real=namespace['multimodal_process']
    def capture_eval(raw_img,vision_shape,img_scores,txt_scores,*args,**kwargs):
        value=real(raw_img,vision_shape,img_scores,txt_scores,*args,**kwargs)
        capture.append({'post_ECI_before_filter':np.array(img_scores,copy=True).reshape(shape),
            'constant_multimodal_activation':float(np.max(np.r_[img_scores,txt_scores[:-1]]))==float(np.min(np.r_[img_scores,txt_scores[:-1]]))})
        return value
    namespace['multimodal_process']=capture_eval
    result=[]
    try:
        # Calling integer round0 returns the FIRST PROMPT map in the author's
        # evaluation API; request each prompt tuple then save the last (answer)
        # map explicitly. This leaves original ECI prompt-history ordering intact.
        for t in range(len(pieces)+1):
            value=fn(tokens,shape,logits,special,vision,processor,'',[0,t],raw_records,True)
        result.append({'round':0,'token_id':tokens[answer],'map':value,
            'raw_CAM':np.array(raw_records[-1],copy=True).reshape(shape),**capture[-1]})
        for r in range(1,rounds):
            value=fn(tokens,shape,logits,special,vision,processor,'',r,raw_records,True)
            result.append({'round':r,'token_id':tokens[answer+r],'map':value,
                'raw_CAM':np.array(raw_records[-1],copy=True).reshape(shape),**capture[-1]})
    finally:
        namespace['multimodal_process']=real
    for row in result:
        if not np.isfinite(row['raw_CAM']).all() or not np.isfinite(row['post_ECI_before_filter']).all():
            raise ValueError('Nonfinite TAM attribution')
    return result
