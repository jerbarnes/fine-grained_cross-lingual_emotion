Cross-lingual approaches to fine-grained emotion detection
==============

We explore cross-lingual aproaches to fine-grained emotion detection
as well as creating fine-grained emotion datasets for Spanish and Catalan
using Best-Worst Scaling.


Requirements to run the experiments
--------
- python3
- sklearn
- scipy
- transformers


Usage
--------

Reports Pearson correlation and p-values for models trained on source language data and tested on source (SRC-SRC) and then tested on target language data (SRC-TRG)

```
python3 bow_regression.py
src_dataset done
trg_dataset done
Training SVR...
SRC-SRC: 0.551 (0.000)
SRC-TRG: 0.285 (0.000)
```

License
-------

Copyright (C) 2019, Jeremy Barnes

Licensed under the terms of the Creative Commons CC-BY public license
