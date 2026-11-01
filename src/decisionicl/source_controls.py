"""Exact matched discovery-source lead interventions."""
from .discovery import build_discovery_prompt
PERSPECTIVES = ('shape and relative proportions', 'color and texture', 'pose and silhouette', 'local visible differences', 'visibility and missing visual information', 'background and environment')
LEADS = {'matched_decision': 'Identify visible factors that you yourself rely on when making classification decisions for this task.', 'generic_visual': 'Identify generic visible properties that describe the supplied images, without prioritizing classification usefulness or your own decision process.', 'task_discriminative': 'Identify visible properties useful for distinguishing the allowed classes in this task, without referring to your own decision process.'}

def matched_source_prompt(task_instruction,mode,labels,perspective):
    shared=build_discovery_prompt(task_instruction=task_instruction,class_names=labels,perspective=perspective,max_groups=6)
    suffix=shared[shared.index('Each factor is one well-defined'):]
    return (LEADS[mode]+'\n\nTask: '+task_instruction+'\nAllowed classes: '+', '.join(labels)+
            '\nPerspective for this pass: '+perspective+'\n\nInspect all supplied TRAIN images. Propose at most 6 visible factor groups. '
            'Seek a different axis from the other perspectives. A factor must be reusable across most images in this task, '
            'visually measurable from pixels, and stated without referring to a class name or sample ID.\n\n'+suffix)
