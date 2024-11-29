
<h1 align="center">CLOnEL</h1>

<div align="center">
<strong>C</strong>ontinual <strong>L</strong>earning <strong>On</strong>tology-enhanced Biomedical <strong>E</strong>ntity <strong>L</strong>inking 
</div>

We propose Ontology-enhanced Entity Linking (OnEL), a novel method that achieves state-of-the-art performance in BioEL by leveraging hierarchical ontology structures for enhanced entity representation. Additionally, we introduce CLOnEL, a framework that leverages continual learning (CL) to validate the effectiveness and broad potential of CL in the BioEL domain. 

## Datasets

### Biomedical Entity Linking (Base)

- [ncbi-disease](https://github.com/dmis-lab/BioSyn)
- [bc5cdr-disease](https://github.com/dmis-lab/BioSyn)
- [bc5cdr-chemical](https://github.com/dmis-lab/BioSyn)
- [COMETA-CF](https://drive.google.com/file/d/1bm_b1dwJYxp3vbMw7vc05-CFWD61JyrF/view?usp=drive_link)
- [AAP](https://drive.google.com/file/d/18VQ6LxSbv8Q4TboTHjeX4DFDAXWd3JLD/view?usp=drive_link)
- [SYMPEL](https://drive.google.com/file/d/1CIzwLaWSq33nKSHe5IS_s_In8LPliWBV/view?usp=drive_link)

### Biomedical Entity Linking (Continual Learning)

[SYMPEL-CL](https://drive.google.com/file/d/1Jyk9_9zKXGCWhx6drxxVTUM4Pd9f9svO/view?usp=drive_link)

### Ontology Tree

- [CTD-Chemical](https://drive.google.com/file/d/1Q8cVl2L-A15sIujKu8e0uu-BZmvHhWqG/view?usp=drive_link)
    - Download from [CTD](https://web.archive.org/web/20180108033447/http://ctdbase.org/downloads)
    - Used for bc5cdr-chemical
- [CTD-Disease](https://drive.google.com/file/d/1BMo38fPwhDWNtb3AHW1GsQFVn8s7dZzD/view?usp=drive_link)
    - Download from [CTD](https://web.archive.org/web/20180108033447/http://ctdbase.org/downloads)
    - Used for bc5cdr-disease and ncbi-disease
- [SNOMED-CT](https://drive.google.com/file/d/1QkqAyZzvknigxQKrAouwLaM0ZiPyFYG-/view?usp=drive_link)
    - Download from [SNOMED-CT](https://www.nlm.nih.gov/healthit/snomedct/index.html?_gl=1*z0twj7*_ga*MzQ4OTkzNTEyLjE2NTYzOTg1Nzc.*_ga_P1FPTH9PL4*MTczMjg1Mzg3Mi40OC4wLjE3MzI4NTM4NzcuMC4wLjA.*_ga_7147EPK006*MTczMjg1Mzg3Mi40Ny4wLjE3MzI4NTM4NzcuMC4wLjA.) 
    - Used for AAP and COMETA
- [ISPO](https://drive.google.com/file/d/1TdvQN6fs3n5oy2SjmRmx8RhgnOdc9lmX/view?usp=drive_link)
    - Used for SYMPEL


## Train

~~~bash
cd OnEL
bash scripts/train/ncbi_disease.sh
~~~
    
    
## Evaluation

~~~bash
cd OnEL
bash scripts/eval/ncbi_disease.sh
~~~    
    
## Results 

### Trained models

- [ncbi-disease](https://huggingface.co/TCMLLM/CLOnEL-NCBI-Disease)
- [bc5cdr-disease](https://huggingface.co/TCMLLM/CLOnEL-BC5CDR-Disease)
- [bc5cdr-chemical](https://huggingface.co/TCMLLM/CLOnEL-BC5CDR-Chemical)
- [cometa-cf](https://huggingface.co/TCMLLM/CLOnEL-COMETA-CF)
- [aap](https://huggingface.co/TCMLLM/CLOnEL-AAP)
- [sympel](https://huggingface.co/TCMLLM/CLOnEL-SYMPEL)

To ensure the fairness of the experiments, we conducted rigorous experiments to validate our approach. The parameter settings followed those specified in the paper, allowing us to reproduce the reported results accurately. However, through a broader search in the parameter space, we discovered that certain parameter configurations could achieve better results than those reported in the paper. For instance, the value of `retrieve_step_ratio` or `tree_ratio` may impacts the results. Similarly, in the AAP task, modifying the `retrieve_func` method to `cosine` yielded a result of 90.1, compared to the reported result of 90.0 in the paper.