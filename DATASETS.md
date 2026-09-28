# Downloaded datasets

All datasets are stored under `datasets/`. The archives are retained in
`datasets/_archives/`, and `scripts/download_datasets.ps1` can resume or verify
the downloads.

| Dataset | Extracted directory | Verified image count | Notes |
| --- | --- | ---: | --- |
| DTD | `datasets/dtd` | 5,640 | Official images, annotations, and ten evaluation splits |
| FGVC Aircraft | `datasets/fgvc-aircraft-2013b` | 10,000 | Official archive; 3,334 train, 3,333 validation, 3,333 test images |
| CUB-200-2011 | `datasets/CUB_200_2011` | 11,788 | Official images and annotations |
| Stanford Dogs | `datasets/stanford_dogs` | 20,580 | Images from a Hugging Face mirror; official Stanford train/test split files; labels and bounding boxes in `metadata.csv` |
| Oxford-IIIT Pet | `datasets/oxford_iiit_pet` | 7,390 raw images | Official archive; annotations define 7,349 samples (3,680 train/validation and 3,669 test) |

## Sources

- DTD: <https://www.robots.ox.ac.uk/~vgg/data/dtd/>
- FGVC Aircraft: <https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>
- CUB-200-2011: <https://www.vision.caltech.edu/datasets/cub_200_2011/>
- Stanford Dogs: <http://vision.stanford.edu/aditya86/ImageNetDogs/>
- Stanford Dogs image mirror: <https://huggingface.co/datasets/dgrnd4/stanford_dog_dataset>
- Stanford Dogs metadata mirror: <https://huggingface.co/datasets/Alanox/stanford-dogs>
- Oxford-IIIT Pet: <https://www.robots.ox.ac.uk/~vgg/data/pets/>

Check each dataset's source page for its license and usage restrictions before
redistribution. In particular, CUB-200-2011 and FGVC Aircraft images are limited
to non-commercial research/educational use by their source terms.

## Re-run

From PowerShell in the repository root:

```powershell
& .\scripts\download_datasets.ps1
```
