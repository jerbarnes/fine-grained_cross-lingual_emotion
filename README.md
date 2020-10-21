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

First, download the pretrained [embeddings](https://drive.google.com/file/d/1GpyF2h0j8K5TKT7y7Aj0OyPgpFc8pMNS/view), unzip them and place the folder called 'embeddings' in the fine-grained_cross-lingual_emotion directory.

The 'models' folder contains the scripts to reproduce the results in the paper.

Reports Pearson correlation and p-values for models trained on source language data and tested on source (SRC-SRC) and then tested on target language data (SRC-TRG)

```
python3 bow_regression.py
src_dataset done
trg_dataset done
Training SVR...
SRC-SRC: 0.551 (0.000)
SRC-TRG: 0.285 (0.000)
```

If you use this code, please cite the following paper:
-------
```
@inproceedings{NavasAlejo2020,
  author =  "Navas Alejo, Irean
        and  Badia, Toni
        and  Barnes, Jeremy",
  title =   "Cross-lingual Emotion Intensity Prediction",
  booktitle =   "Proceedings of the Third Workshop on Computational Modeling of People’s Opinions, Personality, and Emotions in Social Media (PEOPLES 2020)",
  year =    "2020",
  publisher =   "Association for Computational Linguistics",
  pages =   "",
  location =    "Barcelona, Spain"
}
```


License
-------

Copyright (C) 2019, Jeremy Barnes

Licensed under the terms of the Creative Commons CC-BY public license
