# Multilingual Identification of English Code-Switching

This is the code used for the paper "Multilingual Identification of English Code-Switching" (VarDial 2024). I trained two models. One (AnE-LID) that classifies words in a given code-switching text into one of the following classes: `English`, `notEnglish`, `Mixed` and `Other`. And another (AnE-NER) that subcategories these classes if the word is a Named Entity or not (IO scheme).

# Usage

Follow the instructions provided on the Huggingface pages:

AnE-LID: https://huggingface.co/igorsterner/AnE-LID

AnE-NER: https://huggingface.co/igorsterner/AnE-NER

# Reproducibility

To reproduce this work, follow the instructions provided in the READMEs in `data/`. By then, you will have all the required data. You can then train the two models using `train/train-lid.py` and `train/train-ner.py`. Implementations for the baselines and the various evals are also included.

# Citation

Please consider citing this work if it has helped you!

```
@inproceedings{sterner-2024-multilingual,
    title = "Multilingual Identification of {E}nglish Code-Switching",
    author = "Sterner, Igor",
    editor = {Scherrer, Yves  and
      Jauhiainen, Tommi  and
      Ljube{\v{s}}i{\'c}, Nikola  and
      Zampieri, Marcos  and
      Nakov, Preslav  and
      Tiedemann, J{\"o}rg},
    booktitle = "Proceedings of the Eleventh Workshop on NLP for Similar Languages, Varieties, and Dialects (VarDial 2024)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.vardial-1.14",
    doi = "10.18653/v1/2024.vardial-1.14",
    pages = "163--173",
}
```
