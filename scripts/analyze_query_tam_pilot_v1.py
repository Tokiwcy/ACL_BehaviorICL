"""Independent CPU audit and scientific figures for the retrieved TAM pilot."""
import argparse,csv,hashlib,json,math,os,textwrap
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.stats import spearmanr
import query_tam_cloud_ops_v1 as ops
ROOT=Path(__file__).resolve().parents[1]
ARMS=('generic','task','focus_a','focus_b')
LABELS={'generic':'Generic prompt','task':'Original task prompt','focus_a':'Head/beak or texture','focus_b':'Wings/tail or background'}

def measures(a,b):
    a=np.asarray(a,dtype=float).reshape(-1);b=np.asarray(b,dtype=float).reshape(-1)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Aligned finite maps required')
    magnitude=float(np.max(np.abs(a-b)));den=np.linalg.norm(a)*np.linalg.norm(b)
    result={'max_absolute_difference':magnitude,'cosine':float(a@b/den) if den else None,
        'relative_L2_difference':float(np.linalg.norm(a-b)/max(np.linalg.norm(a),np.linalg.norm(b))) if max(np.linalg.norm(a),np.linalg.norm(b)) else 0.,
        'spearman':float(spearmanr(a,b).statistic) if np.ptp(a)>0 and np.ptp(b)>0 else None,
        'all_zero_a':bool(not a.any()),'all_zero_b':bool(not b.any())}
    if np.all(a>=0) and np.all(b>=0) and a.sum()>0 and b.sum()>0:
        p=a/a.sum();q=b/b.sum();m=(p+q)/2
        result['JSD_bits']=float(sum(.5*np.sum(v[v>0]*np.log2(v[v>0]/m[v>0])) for v in (p,q)))
        n=max(1,math.ceil(len(a)*.2));sa=set(np.argsort(a,kind='stable')[-n:]);sb=set(np.argsort(b,kind='stable')[-n:])
        result['top20percent_Jaccard']=len(sa&sb)/len(sa|sb)
    else:result.update(JSD_bits=None,top20percent_Jaccard=None)
    return result

def font(size):
    for p in ('C:/Windows/Fonts/arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(p).exists():return ImageFont.truetype(p,size)
    return ImageFont.load_default()

def overlay(image,map):
    # Fixed original TAM scale 0..255 across all query arms; no panel rescaling.
    shape=(320,250);image=image.copy();image.thumbnail(shape)
    canvas=Image.new('RGB',shape,'white');offset=((shape[0]-image.width)//2,(shape[1]-image.height)//2)
    value=np.array(Image.fromarray(np.uint8(np.clip(map,0,255))).resize(image.size,Image.Resampling.BILINEAR),dtype=float)/255
    # Simple fixed blue-red scientific colour ramp, full 0..255 range.
    colour=np.stack([value,np.zeros_like(value),1-value],-1)*255
    heat=Image.fromarray(np.uint8(colour));canvas.paste(Image.blend(image,heat,.45),offset)
    return canvas

def panel(example,data,output,mode):
    canvas=Image.new('RGB',(1640,580),'white');draw=ImageDraw.Draw(canvas)
    heading=f"{example['dataset'].upper()} {example['sample_id']} | historical ICL: {example['historical_group']} | target: {example['target']}"
    draw.text((15,10),heading,fill='black',font=font(21))
    subtitle='Fixed gold label (teacher-forced, not a prediction)' if mode=='fixed_target' else 'Natural greedy answer; token identities differ across queries'
    draw.text((15,42),subtitle+' | mean of saved token TAMs | common 0..255 scale',fill='#333333',font=font(16))
    image=Image.open(ROOT/example['image_path']).convert('RGB');plain=image.copy();plain.thumbnail((320,250))
    canvas.paste(plain,(15,100));draw.text((15,75),'Original image',fill='black',font=font(18))
    for i,arm in enumerate(ARMS):
        x=335+i*325;record,arrays=data[(arm,mode)]
        canvas.paste(overlay(image,arrays['maps'].mean(0)),(x,100))
        draw.text((x,75),LABELS[arm],fill='black',font=font(18))
        query=example['queries'][arm]
        if arm=='task':query=query.split('The only valid labels:')[0] if 'The only valid labels:' in query else 'Original classify prompt + complete class vocabulary; return one label.'
        lines=textwrap.wrap(query,width=39)
        answer=textwrap.wrap('Output: '+record['output_text'],width=39)
        toks=textwrap.wrap('Mapped tokens: '+' | '.join(record['target_tokens']),width=39)
        draw.multiline_text((x,365),'\n'.join(lines[:4]+answer[:4]+toks[:3]),fill='black',font=font(15),spacing=5)
    draw.text((15,550),'TAM is token visual attribution, not attention weights or causal evidence. Eight-image exploratory pilot.',fill='#444444',font=font(16))
    canvas.save(output)

def analyze(session,output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='-1':raise ValueError('CPU audit only')
    session=Path(session).resolve();output=Path(output).resolve()
    if not session.is_relative_to(ROOT) or not output.is_relative_to(ROOT) or output.exists():raise ValueError('New workspace output and actual independent session required')
    full=ops.read(session/'full_takeback_receipt.json');stop=ops.read(session/'stop_confirmation.json')
    if not full['full'] or not full['closed_archive_member_extracted_SHA_verified'] or (stop['pod_id'],stop['launch_id'])!=(full['pod_id'],full['launch_id']) or (stop['fresh_API']['desiredStatus'],stop['fresh_API']['runtimeStatus'])!=('EXITED','stopped'):raise ValueError('Exact FULL and fresh stopped required before CPU analysis')
    if ops.digest(full['local_archive'])!=full['archive_SHA']:raise ValueError('Local FULL SHA changed')
    snapshot=Path(full['snapshot']);plan=ops.read(ROOT/'analysis/query_tam_pilot_v1/protocol.json')
    if ops.digest(snapshot/'protocol.json')!=ops.digest(ROOT/'analysis/query_tam_pilot_v1/protocol.json'):raise ValueError('Pilot protocol differs')
    for name,expected in full['members'].items():
        if ops.digest(snapshot/name)!=expected:raise ValueError('Extracted FULL member bytes changed')
    complete=ops.read(snapshot/'results/complete.json')
    if complete['protocol_hash']!=plan['protocol_hash'] or complete['runs']!=64:raise ValueError('All 64 declared comparisons required')
    model=ops.read(snapshot/'results/actual_model_environment_SHA.json')
    if model['versions']!=plan['environment'] or model['files_SHA']!=plan['model_inventory']['files_SHA256']:raise ValueError('Original full model/environment proof differs')
    auth=ops.read(snapshot/'auth.json')
    if (auth['pod_id'],auth['launch_id'],auth['protocol_hash'])!=(full['pod_id'],full['launch_id'],plan['protocol_hash']):raise ValueError('Actual source authorization differs')
    for name,expected in auth['sources'].items():
        if ops.digest(snapshot/name)!=expected:raise ValueError('Complete executable/source/input closure differs')
    if ops.digest(snapshot/'tam/tam.py')!=plan['TAM']['SHA']['tam.py']:raise ValueError('Official TAM bytes differ')
    output.mkdir();figures=output/'figures';figures.mkdir()
    comparisons=[];sources={};examples=[]
    for example in plan['examples']:
        data={};pixels=set();grids=set();fixed_ids=set();anchors={}
        for arm in ARMS:
            for mode in ('natural','fixed_target'):
                key=example['dataset']+'_'+example['sample_id']+'_'+arm+'_'+mode
                folder=snapshot/'results/runs'/key;record=ops.read(folder/'record.json')
                if record['key']!=key or record['query']!=example['queries'][arm] or record['image_SHA']!=example['image_SHA'] or ops.digest(folder/'maps.npz')!=record['maps_SHA']:raise ValueError('Exact case/query/image/map identity differs')
                arrays=dict(np.load(folder/'maps.npz',allow_pickle=False))
                if arrays['maps'].shape!=arrays['raw_CAM'].shape or arrays['maps'].shape!=arrays['post_ECI'].shape or arrays['maps'].shape[0]!=len(record['target_ids']) or tuple(arrays['maps'].shape[1:])!=tuple(record['map_shape']) or arrays['anchor_states'].shape!=(36,2560):raise ValueError('Actual spatial/token/layer arrays differ')
                if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite actual saved data')
                pixels.add(record['processed_pixels_SHA']);grids.add(tuple(record['image_grid_thw']))
                if mode=='fixed_target':
                    if not record['fixed_target_is_not_prediction']:raise ValueError('Forced output mislabelled as prediction')
                    if record['output_text']!=example['target']:raise ValueError('Actual forced gold-label output differs')
                    fixed_ids.add(tuple(record['target_ids']));anchors[arm]=arrays['anchor_states']
                sources[key]=record['maps_SHA'];data[(arm,mode)]=(record,arrays)
        if len(pixels)!=1 or len(grids)!=1 or len(fixed_ids)!=1:raise ValueError('Same image/pixels/grid/target-token control failed')
        for left,right in (('generic','task'),('focus_a','focus_b')):
            a=data[(left,'fixed_target')][1];b=data[(right,'fixed_target')][1]
            row={'dataset':example['dataset'],'sample_id':example['sample_id'],'historical_group':example['historical_group'],'pair':left+'_vs_'+right,
                'tokens':data[(left,'fixed_target')][0]['target_tokens']}
            for name in ('maps','raw_CAM','post_ECI'):row[name]=measures(a[name].mean(0),b[name].mean(0))
            row['per_layer_anchor_cosine']=[measures(x,y)['cosine'] for x,y in zip(anchors[left],anchors[right])]
            comparisons.append(row)
        panel(example,data,figures/(example['dataset']+'_'+example['sample_id']+'_fixed.png'),'fixed_target')
        panel(example,data,figures/(example['dataset']+'_'+example['sample_id']+'_natural.png'),'natural')
        # Export every saved token map as a separate image; no visual selection.
        for (arm,mode),(record,arrays) in data.items():
            image=Image.open(ROOT/example['image_path']).convert('RGB')
            for k,map in enumerate(arrays['maps']):overlay(image,map).save(figures/(record['key']+'_token'+str(k)+'.png'))
        examples.append({'dataset':example['dataset'],'sample_id':example['sample_id'],'group':example['historical_group'],'natural_outputs':{arm:data[(arm,'natural')][0]['output_text'] for arm in ARMS}})
    audit={'utc':ops.now(),'independent_CPU_complete_pilot_audit_passed':True,'runs':64,'fixed_images':8,'query_arms':4,
        'protocol_hash':plan['protocol_hash'],'pod_id':full['pod_id'],'launch_id':full['launch_id'],'closed_FULL_SHA':full['archive_SHA'],
        'fresh_exact_stopped_verified':True,'all_source_model_environment_input_token_grid_map_SHA_verified':True,'maps_SHA':sources,
        'comparisons':comparisons,'examples':examples,'interpretation_limits':plan['hypothesis_limits'],
        'original_accuracy_results_not_recomputed':True,'TAM_not_attention_weights_or_causal_proof':True}
    ops.atomic(output/'audit.json',audit)
    summary=['# Fixed-image changed-query TAM pilot','',f"Complete exploratory pilot: 8 images × 4 queries × 2 modes = 64 runs. FULL SHA `{full['archive_SHA']}`; exact cloud GPU stopped before CPU analysis.",'',
        'The original Qwen3-VL-4B-Instruct revision and package versions were reused. All arms have identical processed image bytes and grid. Gold-label probes are teacher-forced and are not predictions.','',
        '| Image | Pair | Raw CAM max difference | TAM cosine | TAM top20% Jaccard | Last-layer anchor cosine |','|---|---|---:|---:|---:|---:|']
    for row in comparisons:
        summary.append('| '+row['sample_id']+' | '+row['pair']+' | '+str(row['raw_CAM']['max_absolute_difference'])+' | '+str(row['maps']['cosine'])+' | '+str(row['maps']['top20percent_Jaccard'])+' | '+str(row['per_layer_anchor_cosine'][-1])+' |')
    summary+=['','## Limits','']+['- '+v for v in plan['hypothesis_limits']]
    (output/'report.md').write_text('\n'.join(summary),encoding='utf-8')
    print(json.dumps({'independent_CPU_pilot_audit_passed':True,'runs':64,'output':str(output)}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--session',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    analyze(args.session,args.output)
