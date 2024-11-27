
<h3 align="center">ClOnEL</h3>

<div align="center">
<strong>C</strong>ontinual <strong>L</strong>earning <strong>On</strong>tology-enhanced Biomedical <strong>E</strong>ntity <strong>L</strong>inking 
</div>

We propose Ontology-enhanced Entity Linking (OnEL), a novel method that achieves state-of-the-art performance in BioEL by leveraging hierarchical ontology structures for enhanced entity representation. Additionally, we introduce CLOnEL, a framework that leverages continual learning (CL) to validate the effectiveness and broad potential of CL in the BioEL domain. 

## Datasets

### Biomedical Entity Linking (Base)

ncbi-disease

bc5cdr-disease

bc5cdr-chemical

COMETA-CF

AAP

SYMPEL



### Biomedical Entity Linking (Continual Learning)

SYMPEL-CL

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
- [COMETA-CF](https://huggingface.co/TCMLLM/CLOnEL-COMETA-CF)
- [AAP](https://huggingface.co/TCMLLM/CLOnEL-AAP)

To ensure the fairness of the experiments, we conducted rigorous experiments to validate our approach. The parameter settings followed those specified in the paper, allowing us to reproduce the reported results accurately. However, through a broader search in the parameter space, we discovered that certain parameter configurations could achieve better results than those reported in the paper. For instance, the value of `retrieve_step_ratio` or `tree_ratio` may impacts the results. Similarly, in the AAP task, modifying the `retrieve_func` method to `cosine` yielded a result of 90.1, compared to the reported result of 90.0 in the paper.