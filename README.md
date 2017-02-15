# BDiF

Instructions:

1) Build program: python setup.py build_ext --inplace

2) Scrub:
    a. Modify scrub.cfg
        -- data_file: raw data file
        -- noise_file: output noise file
        -- memory per iteration: how much memory each time the program read in from file
        -- ticks per iteration: how many lines to process per loop

    b. Run program: ./scrub.sh

3) Normal:
    a. Modify normaltest.cfg
    b. Run program: ./normal.sh


Ideas:

1) Scrub:
    I am filtering out the following ticks:
    a. Invalid data points including negative price/volume, incorrect time (previous day etc).
    b. Outliers which passes the return threshold which is a parameter in scrub.cfg

2) Normal:
    For normal test, for every chunk (determined by ticks per iteration), I will perform
    normal test and return a p_value. In the end the program will output how many samples 
    passed the normal test. 
