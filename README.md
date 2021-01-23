# Machine_Learning_Project2

For this project I will apply Logistic Regression to predict whether capacitors from a fabrication plant
pass quality control based (QC) on two different tests. To train my system and determine its reliability
there is a total of 118 example sets.
I have already randomized the data into two data
sets: a training set of 83 examples and a test set
of 35 examples. Both are formatted as
•First line: m and n, tab separated
•Each line after that has two real numbers
representing the results of the two tests,
followed by a 1.0 if the capacitor passed QC and
a 0.0 if it failed QC—tab separated.
I created a binary classifier to
predict whether each capacitor in the test set will pass QC. Since this data is
not linear, I added new features based on powers of the original two features. 
