# GoogleQuestPJ

School of Computer Science, Faculty of Exact Sciences, Tel-Aviv University
https://en-exact-sciences.tau.ac.il/computer

* 2019-2020 Data Science Workshop 
* As part of the Workshop, our group (Maintainers) took the Kaggle's Google QUEST Q&A Labelling challenge:

Kaggle - Google QUEST Q&A Labeling:
https://www.kaggle.com/c/google-quest-challenge

The challenge: 
* using Google’s CrowdSource team dataset, build predictive algorithms for different subjective aspects of question-answering
* Improving automated understanding of complex question answer content

## Requirements

* [Python 3.X](https://docs.python.org/3/)
* [pip](https://pip.pypa.io/en/stable/installing/)

Python Package dependencies listed in [requirements.txt](requirements.txt)

## Getting Started

```bash
$ git clone https://github.com/meniomaisy/GoogleQuestPJ
$ cd GoogleQuestPJ
$ pip install -r requirements.txt
```

_Note:_ The above installation downloads the best-matching default english language model for spaCy. But to improve the model's accuracy you can install other models too. Read more at [spaCy docs](https://spacy.io/usage/models).

```bash
$ python -m spacy download en_core_web_md
```

## Malfunction

ModuleNotFoundError: No module named 'SOME_MODULE':

* the problem is probably related with previous version of the module, try:
	```bash
	$ pip install SOME_MODULE --user
	```
* or if module exist but not working properly:
	```bash
	$ pip install --upgrade SOME_MODULE >= ( OR == ) VERSION_BY_REQUIERMENTS --user
	```	

## Running

Next step is to enter the python Notebook and run:

Q&A-GoogleQuest.ipynb ## via Jupyter Notebook

## References

For full project documentation please read:
* Q&A Google quest Documentation.docx

### Features

* Classify questions with regular expression (default)
* Classify questions with a SVM (optional)

### Maintainers
* Anav Shapira
* Mandy Rosemblaum
* Meni Omaisy
* Eliran Eitan
