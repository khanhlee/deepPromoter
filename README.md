# FastSpot
## A sequence-based approach for identifying recombination spots in Saccharomyces cerevisiae by using hyper-parameter optimization in FastText and support vector machine

### Step 1
Install FastText package via the instructions here: https://github.com/facebookresearch/fastText

### Step 2
Use "fasttext_generated.py" file to transform FASTA sequence into FastText format
- *python fasttext_generated.py fasta_file fasttext_file*

### Step 3
Print vectors using FastText model:
- *fasttext print-sentence-vectors model.bin < fasttext_file > vector_file*

### Step 4
Use "linux_svm.py" to predict the generated file:
- *python linux_svm.py spot.wN5.cv.csv vector_file output_file*

### Step 5
Check in *output_file* for the result:
- '1' is hotspot
- '0' is coldspot
