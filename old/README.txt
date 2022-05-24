All functional scripts that were used previously but now are not useful or suitable.

'cross_talk_rate' and 'cross_talk_rate_bin' were used as a standalone scripts, with 
modules 'zeros_to_valid' and 'zeroes_to_valid_bin' for calculating cross-talk rates based on the data from '.txt' and '.dat' files, respectively. All four scripts were combined into
a single module, which is called in the main.py script.

'differences' used to test timestamp differences between all pixels, but only in a single slice of data lines, therefore is heavily flawed.