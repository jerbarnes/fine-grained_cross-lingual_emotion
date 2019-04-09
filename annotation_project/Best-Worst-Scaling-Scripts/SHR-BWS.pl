#!/usr/bin/perl

################################################################################################################################
#  Author: Svetlana Kiritchenko
#  Information and Communications Technologies / Technologies de l'information et des communications
#  National Research Council Canada /Conseil national de recherches Canada
#  
#  Description: calculates split-half reliability for Best-Worst Scaling annotations.
#               All available annotations per tuple are split randomly in two half-sets, 
#               the scores for all items are calculated independently from the two half-sets, and
#               the correlation between these two sets of scores is reported.  
#
#  Usage: SHR-BWS.pl <file-annotations>
#    <file-annotations> is a CSV (comma delimitted) file with item tuples and Best-Worst annotations.
#
#  Output: average Spearman rank correlation and Pearson correlation for the scores obtained from two annotation half-sets.
#    The output is written into STDOUT. 
#  
#  To run the script, you need the packages Text::CSV, Statistics::Basic, Statistics::RankCorrelation to be installed.
#
#  Version: 1.1
#  Last modified: May 30, 2017
#
#################################################################################################################################

use warnings;
use strict;
use utf8;
binmode (STDIN,  ":utf8");
binmode (STDOUT, ":utf8");

use Text::CSV;
use List::Util qw(shuffle);
use Statistics::Basic qw (vector mean stddev);
use Statistics::RankCorrelation;


#################################################################################################################################
# PARAMETERS 
#################################################################################################################################

# number of test trials
my $num_trials = 100;


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

# random number seed (to make results reproducible)
my $rand_seed = 1234;
srand($rand_seed);


# file with the BWS annotations
my $file_annotations = $ARGV[0];

# read the annotation file
my %ann = ();
read_annotation_file($file_annotations, \%ann);


my @corr_spearman = (), my @corr_pearson = ();
for(my $trial = 0; $trial < $num_trials; $trial++) {

	# split data into two non-intersecting sets of annotations
	my @set1 = (), my @set2 = (); 
	get_random_ann(\%ann, \@set1, \@set2);

	# calculate the scores on both sets
	my %scores1 = (), my %scores2 = ();
	get_scores_counting(\@set1, \%scores1);
	get_scores_counting(\@set2, \%scores2);

	# compare scores on the two sets
	(my $spearman_i, my $pearson_i) = correlation(\%scores1, \%scores2);	
	push(@corr_spearman, $spearman_i);
	push(@corr_pearson, $pearson_i);
}

my $spearman_vec = vector(@corr_spearman);
printf "SHR Spearman correlation: %.4f +/- %.4f\n", mean($spearman_vec), stddev($spearman_vec);

my $pearson_vec = vector(@corr_pearson);
printf "SHR Pearson correlation: %.4f +/- %.4f\n", mean($pearson_vec), stddev($pearson_vec);



# read the Best-Worst annotation file
sub read_annotation_file {
	my($file_annotations, $ann) = @_;
	
	print STDERR "Reading the annotation file $file_annotations ...\n";

	my $csv = Text::CSV->new ( { binary => 1 } )  
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
			$items_in_tuple{$item} = 1;
		}

		my @items_in_tuple_ordered = sort keys %items_in_tuple;
		my $tuple_string = join("!***###***!", @items_in_tuple_ordered);
		
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

		push(@{$ann->{$tuple_string}->{best}}, $best_item);
		push(@{$ann->{$tuple_string}->{worst}}, $worst_item);
		
		
		$line_num++;  
	}	
		
	close($fh);

	print STDERR "Read ".($line_num - 1)." annotations.\n";
}


# split annotations for each tuple randomly into two half-sets
sub get_random_ann {
	my($ann, $set1, $set2) = @_;
	
	foreach my $t (sort keys %{$ann}) {
		my @items_in_tuple = split(/\!\*\*\*\#\#\#\*\*\*\!/, $t);
		
		my $total_ann = scalar(@{$ann->{$t}->{best}});
		
		# number of annotations in a half-set
		my $num_selected = int($total_ann/2);
		# if the total number of annotations is an odd number (N),
		# randomly select int(N/2) or int(N/2)+1 for the number of annotations in the first half-set
		if((($total_ann % 2) != 0) && (rand() < 0.5)) {
			$num_selected++;
		}

		my @ann_index = randomize_index($total_ann);	
	
		my $j;
		# annotations in the first half-set
		for($j = 0; $j < $num_selected; $j++) {
			push(@{$set1->[@{$set1}]}, @items_in_tuple, $ann->{$t}->{best}->[$ann_index[$j]], $ann->{$t}->{worst}->[$ann_index[$j]]);
		}

		# annotations in the second half-set
		for(; $j < @ann_index; $j++) {
			push(@{$set2->[@{$set2}]}, @items_in_tuple, $ann->{$t}->{best}->[$ann_index[$j]], $ann->{$t}->{worst}->[$ann_index[$j]]);
		}
	}
}


# calculating the scores for the items
# with Count method
sub get_scores_counting {
	my($set, $scores) = @_;
	
	my %count_item = (), my %count_best = (), my %count_worst = ();
	
	for(my $i = 0; $i < @{$set}; $i++) {
		# counting occurrences of items in a tuple
		for(my $j = 0; $j < (@{$set->[$i]} - 2); $j++) {
			$count_item{$set->[$i]->[$j]}++;
		}
		
		# counting items selected as best and worst
		$count_best{$set->[$i]->[-2]}++;
		$count_worst{$set->[$i]->[-1]}++;
	}
	
	# calculating the scores for all items
	foreach my $item (keys %count_item) {
		if(!defined $count_best{$item}) { $count_best{$item} = 0; }
		if(!defined $count_worst{$item}) { $count_worst{$item} = 0; }
		
		$scores->{$item} = ($count_best{$item} - $count_worst{$item}) / $count_item{$item};
	}
}


# calculate the correlation between two sets of scores
sub correlation {
	my($scores1, $scores2) = @_;
	
	my @ratings1 = (), my @ratings2 = ();

	foreach my $term (keys %{$scores1}) {
		push(@ratings1, $scores1->{$term});	
		push(@ratings2, $scores2->{$term});
	}

	# calculating Spearman rank correlation
	my $cor = Statistics::RankCorrelation->new(\@ratings1, \@ratings2);
	my $rho = $cor->spearman;

	# calculating Pearson correlation
	my $pearson = pearson_correlation(\@ratings1, \@ratings2);
	
	return($rho, $pearson);
}

# calculating mean for a set of values
sub my_mean {
	my($x) = @_;
	
	my $n = scalar(@{$x});
	
	my $sum = 0;
	for(my $i = 0; $i < $n; $i++) {
		$sum += $x->[$i];
	}
	
	return $sum/$n;
}

# calculating covariance for two sets of values
sub my_cov {
	my($x, $y) = @_;
	
	my $x_mean = my_mean($x);
	my $y_mean = my_mean($y);
	
	my $sum = 0;
	for(my $i = 0; $i < @{$x}; $i++) {
		$sum += ($x->[$i] - $x_mean) * ($y->[$i] - $y_mean);
	}
	
	return $sum;
}

# calculating Pearson correlation
sub pearson_correlation {
	my($x, $y) = @_;
	
	return my_cov($x,$y)/(sqrt(my_cov($x,$x))*sqrt(my_cov($y,$y)));
}


# shuffle integers from 0 to $n
sub randomize_index {
	my($n) = @_;
	
	my @index = ();
	for(my $i = 0; $i < $n; $i++) {
		push(@index, $i);
	}
	
	my @random_index = shuffle(@index);
	
	return(@random_index);
}

