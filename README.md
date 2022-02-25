Data and Code for the paper XXXXX.

## Folder Structure

- `data/`: Tweet IDs for `tusc-city` and `tusc-country` subsets. Each folder is split by city/country, and further by month. Each file contains the ID of the tweet. 
- `code/`: Code for running the emotion dynamics and emotion word usage analyses.
    - `clean_data.py`: functions to tokenize tweets and identify URLs.
    - `avgEmoValues.py`: functions to obtain number of tokens per tweet, number of lexicon tokens per tweet, and the average emotion value of words in that tweet, given an emotion lexicon.
    - `uedLib`/: code to run emotion dynamics on tweet data. 
        - `run_ued.py`: sample call to the main UED library.
        - `lib/ued.py`: main UED module that takes a config file as input and outputs emotion dynamic metrics.
        - `config/`: sample config files for running UED on (a) a single file; (b) a folder with multiple CSV files.

- `lexicons/`: Files with words and associated values for different emotion dimensions. Please see terms of use for these lexicons from their original source at http://saifmohammad.com/WebPages/lexicons.html. This folder contains three main categories of lexicons for each of the valence-arousal-dominance dimensions:
    - `<dim>.csv`: The complete lexicon
    - `<dim>_polar.csv`: Does not contain words with a score between (0.33, 0.67).
    - `<dim>_low.csv`: Terms with scores <=0.33
    - `<dim>_high.csv`: Terms with scores >=0.67

    All lexicon files have two columns: `word` and `val`.