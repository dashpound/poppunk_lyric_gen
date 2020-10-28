# Pop Punk Lyric Generator

This is the full workflow for producing a pop punk lyric generator.

## Repo Structure

The folders are organized by function.

* The "scraper_bot" folder is configured to pull an artist from AZlyrics
* The "preprocessing" folder is designed to perform word2vec and TF-IDF preprocessing scraper bot results
* The "rnn" folder is the actual deep learning model and weightings that produce pop punk lyrics
* The "end_to_end" folder contains a script that calls all the required scripts to generate results.
* The "end_to_end" script is the key script for the full data pipeline.

scraper_bot -> preprocessing -> rnn 

All called by run_all.py in the end_to_end folder.

### Prerequisites

The following packages will need to be installed in order to run the project code.

```
python3.7.x
nltk
gensim.models.doc2vec
tensorflow
keras
```

## Contributing

* John Kiley

## Acknowledgments

* Taylor Swift Lyric Generator 
