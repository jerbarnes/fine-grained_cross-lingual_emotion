Best Worst Scaling Annotator
==============

A GUI to anntotate for best worst scaling. It takes as input a file where each line has 4 instances divided by tabs. It then displays these 4-tuples and allows you to annotate them as the one that most displays the emotion (1-click, button is red), the instance that least represents the emotion (2-clicks, button is blue), or neither (button is grey, default). The export button lets you save the annotations as a json file in ./anns.


Requirements to run the annotator
--------
python 3



Usage
--------
python3 annotator.py  optional: [--emotions love hate] [--annotator_name Jeremy]


License
-------

Copyright (C) 2019, Jeremy Barnes

Licensed under the terms of the Creative Commons CC-BY public license
