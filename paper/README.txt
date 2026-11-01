DecisionICL CVPR LaTeX source

1. Upload this ZIP to Overleaf.
2. Select main.tex for the paper or supplement.tex for the supplement.
3. Use pdfLaTeX and BibTeX, or the default Overleaf LaTeX build.
4. Replace paperID ***** in both TeX files with the real submission ID.

All tables are included directly in the two TeX files.
figures/ contains only the four required PDF figures.
No model, dataset, Python script, audit record, or extra download is needed to compile.

The main accuracy table combines five datasets and three target models.
At each budget the RICES group contains RICES base, +MUIER, +DecisionICL,
and +MUIER+DecisionICL. CIRCLES remains a separate two-shot group.
The +MUIER rows retain the recorded SigLIP classification adaptation.
RICES base retains CLIP. This grouping changes presentation only.
The main paper uses the original accepted seed42 results for all target models.
All three-run values, means, and sample SDs are confined to the supplement.
The supplement retains seeds 42, 17, and 73 and their paired evidence.
It also includes all 90 accepted VICL/Prompt-SelF auxiliary method results.
MMICES question-only retrieval is documented as plan-equivalent to RICES.

Terminology uses class selection for the argmax of fused retrieval scores.
Its class index identifies the training pool for demonstration retrieval.
The final VLM answer is generated after demonstrations and query enter the prompt.
The workflow and case figures show these as separate stages.
Main-text discussion also quantifies selected-class readout versus final answers.

Template source is cvpr-org/author-kit, CVPR2026-v1(latex), in anonymous review mode.
Target conference metadata is CVPR 2027.
