wScripts for Best-Worst Scaling
Version 1.2
25 April 2017
Copyright (C) 2016 National Research Council Canada (NRC)
Contact: Saif Mohammad (saif.mohammad@nrc-cnrc.gc.ca)


*********************************************************************************
Terms of use
*********************************************************************************

1. These scripts can be used freely for research purposes.
2. If you use the scripts, then please cite the associated papers:

Svetlana Kiritchenko and Saif M. Mohammad (2016) Capturing Reliable Fine-Grained Sentiment Associations by Crowdsourcing and Best-Worst Scaling. Proceedings of the 15th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL), San Diego, California, 2016.

Svetlana Kiritchenko and Saif M. Mohammad (2017) Best-Worst Scaling More Reliable than Rating Scales: A Case Study on Sentiment Intensity Annotation. Proceedings of the Annual Meeting of the Association for Computational Linguistics, Vancouver, Canada, 2017.

3. If interested in commercial use of the scripts, send email to the contact.
4. If you use the scripts in a product or application, then please credit the authors and NRC appropriately. Also, if you send us an email, we will be thrilled to know about how you have used the scripts.
5. National Research Council Canada (NRC) disclaims any responsibility for the use of the scripts and does not provide technical support. However, the contact listed above will be happy to respond to queries and clarifications.
6. Rather than redistributing the scripts, please direct interested parties to this page:
   http://www.saifmohammad.com/WebPages/BestWorst.html


Please feel free to send us an email:
- with feedback regarding the scripts;
- with information on how you have used the scripts;
- if interested in having us analyze your data for sentiment, emotion, and other affectual information;
- if interested in a collaborative research project.



*********************************************************************************
General Description
*********************************************************************************

Best–Worst Scaling (BWS), also sometimes referred to as Maximum Difference Scaling (MaxDiff), is an annotation scheme that exploits the comparative approach to annotation (Louviere and Woodworth, 1990; Cohen, 2003; Louviere et al., 2015). Annotators are given k items (k-tuple) and asked which item is the Best (highest in terms of the property of interest) and which is the Worst (lowest in terms of the property of interest). K typically ranges from 4 to 5. These annotations can then be easily converted into real-valued scores of association between the items and the property, which eventually allows for creating a ranked list of items as per their association with the property of interest.

- We have used Best-Worst Scaling to manually annotate words and phrases for sentiment through crowdsourcing. We have shown that ranking of terms by sentiment remains remarkably consistent even when the annotation process is repeated with a different set of annotators. The details of this project can be found in the following paper:

Svetlana Kiritchenko and Saif M. Mohammad (2016) Capturing Reliable Fine-Grained Sentiment Associations by Crowdsourcing and Best-Worst Scaling. Proceedings of the 15th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL), San Diego, California, 2016.

- We have also compared the reliability of annotations produced with BWS and the reliability of annotations obtained with conventional rating scales. We showed that BWS annotations are significantly more reliable, especially when only a limited number of annotations (up to 5N, where N is the number of items to be rated) can be obtained. The details of this study can be found in the following paper:

Svetlana Kiritchenko and Saif M. Mohammad (2017) Best-Worst Scaling More Reliable than Rating Scales: A Case Study on Sentiment Intensity Annotation. Proceedings of the Annual Meeting of the Association for Computational Linguistics, Vancouver, Canada, 2017.

- The sentiment lexicons annotated with Best-Worst Scaling and the papers describing their creation and use can be found at http://www.saifmohammad.com/WebPages/BestWorst.html.



*********************************************************************************
The Current Package
*********************************************************************************

The current package includes three scripts and four example files.

Main scripts:
  - "generate-BWS-tuples.pl" is a Perl script that generates tuples for Best-Worst Scaling annotation from a given list of items/terms;
  - "get-scores-from-BWS-annotations-counting.pl" is a Perl script that converts Best-Worst annotations into real-valued scores of association of items with the property of interest.

Additional scripts:
  - "SHR-BWS.pl" is a Perl script that calculates split-half reliability (SHR) of BWS annotations.


Example files:
  - "example-items.txt" is an example file with a list of items; it can be used as an example input file for "generate-BWS-tuples.pl";
  - "example-items.txt.tuples" is an example output file for "generate-BWS-tuples.pl";
  - "example-tuples-annotations.csv" is an example annotation file; it can be used as an example input file for "get-scores-from-BWS-annotations-counting.pl";
  - "example-scores.txt" is an example output file for "get-scores-from-BWS-annotations-counting.pl".



*********************************************************************************
Script for generating item tuples (generate-BWS-tuples.pl)
*********************************************************************************

This script generates tuples for Best-Worst Scaling annotation from a given list of items/terms. Each tuple has exactly k items, where k is usually 4 or 5 and can be set up by the user. Another parameter that can be set up by the user is the number of tuples to generate. In practice, around 1.5 x N to 2 x N tuples (where N is the total number of items) are sufficient to obtain reliable scores.

The tuples are generated by random sampling and satisfy the following criteria:
1. no two items within a tuple are identical;
2. each item in the item list appears approximately in the same number of tuples;
3. each pair of items appears approximately in the same number of tuples.

The script generates many sets of tuples and outputs the one that best satisfies the above criteria. The number of iterations can be set up by the user.

All parameters can be set directly in the script. Below is the full list of changeable parameters.


***** PARAMETERS: *****

$items_per_tuple: number of items per tuple (typically, 4 or 5)

$factor: Best-Worst Scaling factor (typically 1.5 or 2); multiply the total number of items by this factor in order to determine the number of tuples to generate

$num_iter: number of iterations (typically 100 or 1000)


***** USAGE: *****

generate-BWS-tuples.pl <file-items>

where <file-items> is a file that contains a list of items to be annotated (one item per line)

Output: a list of item tuples (one tuple per line; items in a tuple are separated by tab). The output is written into file <file-items>.tuples.

Example usage: generate-BWS-tuples.pl example-items.txt
The output file should look similar to example-items.txt.tuples.




*********************************************************************************
Script for calculating scores (get-scores-from-BWS-annotations-counting.pl)
*********************************************************************************

This script converts Best-Worst annotations into real-valued scores of association of items with the property of interest. There are several ways of doing the conversion. This scripts implements the simplest and fastest procedure called Counts Analysis (Orme, 2009): For each item, its score is calculated as the percentage of times the item was chosen as the Best minus the percentage of times the item was chosen as the Worst. The scores range from -1 (least association with the property of interest) to 1 (most association with the property of interest).


***** USAGE: *****

get-scores-from-BWS-annotations-counting.pl <file-annotations>

where <file-annotations> is a CSV (comma delimitted) file with item tuples and Best-Worst annotations. Each line should contain all k items from the tuple, and the items annotated as the Best and as the Worst. For example,

item1,item2,item3,item4,best,worst

File "example-tuples-annotations.csv" shows an example. The annotation file can contain other columns, but they will be ignored. The script assumes that the file has the column names as the first line, and there are columns named "Item1", "Item2", "Item3", "Item4" (for tuple items), and "BestItem", "WorstItem" (for annotations). The user can provide their own column names by modifying the @column_names parameter in the script. Also, there can be more or less than 4 item columns, but the column names for all items should precede the column names for the Best and Worst annotations in @column_names (e.g., "Item1", "Item2", "Item3", "Item4", "Item5", "BestItem", "WorstItem").

Output: items with scores (one item per line). The output is written into STDOUT.

Example usage: get-scores-from-BWS-annotations-counting.pl example-tuples-annotations.csv
The output should look similar to example-scores.txt.




*********************************************************************************
Script for calculating split-half reliability (SHR-BWS.pl)
*********************************************************************************

This script calculates split-half reliability (SHR) of BWS annotations over a number of trials. SHR is a commonly used approach to determine consistency in psychological studies, that we employ as follows. All annotations for a tuple are randomly split into two halves. Two sets of scores are produced independently from the two halves. Then the correlation between the two sets of scores is calculated. If a method is more reliable, then the correlation of the scores produced
by the two halves will be high.


***** PARAMETERS: *****

$num_trials: number of test trials (typically 100)


***** USAGE: *****

SHR-BWS.pl <file-annotations>

where <file-annotations> is a CSV (comma delimitted) file with item tuples and Best-Worst annotations. Each line should contain all k items from the tuple, and the items annotated as the Best and as the Worst. For example,

item1,item2,item3,item4,best,worst

File "example-tuples-annotations.csv" shows an example. The annotation file can contain other columns, but they will be ignored. The script assumes that the file has the column names as the first line, and there are columns named "Item1", "Item2", "Item3", "Item4" (for tuple items), and "BestItem", "WorstItem" (for annotations). The user can provide their own column names by modifying the @column_names parameter in the script. Also, there can be more or less than 4 item columns, but the column names for all items should precede the column names for the Best and Worst annotations in @column_names (e.g., "Item1", "Item2", "Item3", "Item4", "Item5", "BestItem", "WorstItem").

Output: average Spearman rank correlation and Pearson correlation for the scores obtained from two annotation half-sets. The output is written into STDOUT.



*********************************************************************************
More Information
*********************************************************************************

Svetlana Kiritchenko and Saif M. Mohammad (2016) Capturing Reliable Fine-Grained Sentiment Associations by Crowdsourcing and Best-Worst Scaling. Proceedings of the 15th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL), San Diego, California, 2016.

Svetlana Kiritchenko and Saif M. Mohammad (2017) Best-Worst Scaling More Reliable than Rating Scales: A Case Study on Sentiment Intensity Annotation. Proceedings of the Annual Meeting of the Association for Computational Linguistics, Vancouver, Canada, 2017.


Steven H. Cohen. (2003) Maximum difference scaling: Improved measures of importance and preference for segmentation. Sawtooth Software, Inc.

Jordan J. Louviere and George G. Woodworth (1990) Best-worst analysis. Working Paper. Department of Marketing and Economic Analysis, University of Alberta.

Jordan J. Louviere, Terry N. Flynn, and A. A. J. Marley (2015) Best-Worst Scaling: Theory, Methods and Applications. Cambridge University Press.

Bryan Orme (2009) Maxdiff analysis: Simple counting, individual-level logit, and HB. Sawtooth Software, Inc.

