# DeepPromoter
## A sequence-based approach for classifying DNA promoters by using deep learning and continuous FastText N-gram levels

### Step 1
Install FastText package via the instructions here: https://github.com/facebookresearch/fastText

### Step 2
Use "fasttext_generated.py" file to transform FASTA sequence into FastText format
- *python fasttext_generated.py fasta_file fasttext_file*

### Step 3
Print vectors using FastText model:
- *fasttext print-sentence-vectors model.bin < fasttext_file > vector_file*

### Step 4
Use "promoter_cnn.py" to train and evaluate  model based on generated vectors:
- *python promoter_cnn.py vector_file*

## Citation
Please cite our paper as:
>Le NQK, Yapp EKY, Nagasundaram N and Yeh H-Y (2019) Classifying Promoters by Interpreting the Hidden Information of DNA Sequences via Deep Learning and Combination of Continuous FastText N-Grams. *Front. Bioeng. Biotechnol.* 7:305. doi: 10.3389/fbioe.2019.00305
