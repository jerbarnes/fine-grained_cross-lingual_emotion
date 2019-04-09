#!/usr/bin/perl

################################################################################################################################
#  Authors: Svetlana Kiritchenko, Peter Turney
#  Information and Communications Technologies / Technologies de l'information et des communications
#  National Research Council Canada /Conseil national de recherches Canada
#  
#  Description: calculates real-valued scores for items annotated with Best-Worst Scaling method;
#               uses the simple counting procedure: Best - Worst. 
#
#  Usage: get-scores-from-BWS-annotations-counting.pl <file-annotations>
#    <file-annotations> is a CSV (comma delimitted) file with item tuples and Best-Worst annotations.
#
#  Output: items with scores (one item per line) 
#    The output is written into STDOUT. 
#  
#  To run the script, you need the package Text::CSV to be installed.
#
#  Version: 1.0
#  Last modified: June 28, 2016
#
#################################################################################################################################

use warnings;
use strict;
use utf8;
binmode (STDIN,  ":utf8");
binmode (STDOUT, ":utf8");

use Text::CSV;


#################################################################################################################################
# Names of the columns in the annotation data.
# PLEASE MODIFY AS NEEDED!
#
# There can be more or less than 4 item columns, 
# but the column names for all items should precede the column names for the Best and Worst annotations 
# (e.g., "Item1", "Item2", "Item3", "Item4", "Item5", "BestItem", "WorstItem").
#################################################################################################################################

my @column_names = ("Item1", "Item2", "Item3", "Item4", "BestItem", "WorstItem");

#################################################################################################################################


die "Usage: get-scores-from-BWS-annotations-counting.pl <file-annotations>\n" if(@ARGV < 1);

# file with the BWS annotations
my $file_annotations = $ARGV[0];


# read the Best-Worst annotation file
print STDERR "Reading the annotation file $file_annotations ...\n";

my %count_item = (); my %count_best = (); my %count_worst = ();

my $csv = Text::CSV->new ( { binary => 1 } )  # should set binary attribute.
                 or die "Cannot use CSV: ".Text::CSV->error_diag ();
 
open my $fh, "<", $file_annotations or die "Cannot open the annotation file $file_annotations: $!";

# read the the first line with the column names
# check if all @column_names can be found
my $column_names_row = $csv->getline($fh);
my %data_column_names = ();
foreach my $col (@{$column_names_row}) {
	$data_column_names{$col} = 1;
}

foreach my $col (@column_names) {
	if(!defined $data_column_names{$col}) {
		print STDERR "ERROR: Cannot find a column named $col.\n";
		exit();
	}
}

$csv->column_names ($column_names_row);

# read the annotations line by line
my $line_num = 1;
while (my $data = $csv->getline_hr($fh)) {
	
	# read the items
	my %items_in_tuple = ();
	for(my $i = 0; $i < (@column_names - 2); $i++) {
		my $item = $data->{$column_names[$i]};
		$count_item{$item}++;
		$items_in_tuple{$item} = 1;
	}
	
	# read the Best and Worst items
	my $best_item = $data->{$column_names[-2]};
	my $worst_item = $data->{$column_names[-1]};
	
	# check if the Best and Worst items are among the tuple items
	if(!defined $items_in_tuple{$best_item}) {
		print STDERR "ERROR: Illegible annotation for the Best item in line $line_num.\n";
		exit();
	}
	if(!defined $items_in_tuple{$worst_item}) {
		print STDERR "ERROR: Illegible annotation for the Worst item in line $line_num.\n";
		exit();
	}

	# check if the Best and Worst items are the same
	if($best_item eq $worst_item) {
		print STDERR "WARNING: Annotations for the Best and Worst items are identical in line $line_num.\n";
	}
	
	$count_best{$best_item}++;
	$count_worst{$worst_item}++;

	$line_num++;  
}	
	
close($fh);

print STDERR "Read ".($line_num - 1)." annotations.\n";
print STDERR "Found ".scalar(keys %count_item)." unique items.\n";


# calculating the scores for the items
my %scores = ();
foreach my $item (keys %count_item) {
	if(!defined $count_best{$item}) { $count_best{$item} = 0; }
	if(!defined $count_worst{$item}) { $count_worst{$item} = 0; }
	
	$scores{$item} = ($count_best{$item} - $count_worst{$item}) / $count_item{$item};
}

# writing the scores to STDOUT
print STDERR "\nWriting the scores to STDOUT ...\n";

foreach my $item (sort {$scores{$b} <=> $scores{$a}} keys %scores) {
	printf "%s\t%.3f\n", $item, $scores{$item};
}

print STDERR "Finished.\n";



