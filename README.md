# scholar
This code provides an interface for Word2Vec in Python. It uses a part-of-speech-tagged corpus, so that we can query for specific parts of speech. It is intended to be used with [autoplay](https://github.com/danielricks/autoplay), a learning environment for interactive fiction.

## Requirements

Per standard natural language processing, we used [Penn Treebank tags](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html) to denote parts of speech. We tagged a copy of Wikipedia from January 2016 (using [Parsey McParseface](https://github.com/tensorflow/models/tree/master/syntaxnet)) and used that as input to word2vec.

Our processed files are available [here](https://drive.google.com/open?id=0B3lpCS07rg43bVBmd1lSVUVSb28). We provide a truncated version of the binary file as well, stored in pickle format. This truncated version contains the top 30,000 most popular singular nouns on Wikipedia, and the top 3,000 verbs. It runs much faster than the full binary, but also loses access to some methods. I would honestly recommend using the truncated corpus, since the speed-up is significant, and most of the real functionality is included. That download link also provides pre-computed distributions of word-to-pos-tags, to speed up computation during run-time.

## Usage

Using linear algebra, we were able to pull affordances (relevant verbs) for nouns out of word2vec, with varying success on other parts of speech. This interface provides methods that perform those operations.

The available methods using the full binary are demonstrated below. Some methods require a pos-tag, and some of them don't.

```python
import scholar.scholar as sch

# This will load word2vec using the full corpus
s = sch.Scholar()

# These methods require a Penn Treebank tag

s.get_cosine_similarity('man_NN', 'woman_NN')
s.analogy('king_NN -man_NN woman_NN')
s.exists_in_model('peppercorn_NN')

# The below methods DO NOT require the use of a Penn Treebank tag.

# For singular nouns
s.get_verbs('mailbox', 10) # Where 10 is the number of desired results
s.get_adjectives('mailbox', 10)
s.get_hypernyms('tree', 10)
s.get_hyponyms('weapon', 10)
s.get_parts('house', 10)
s.get_whole('bread', 10)

# For plural nouns
s.get_verbs_plural('mailboxes', 10)
s.get_adjectives_plural('mailboxes', 10)
s.get_hypernyms_plural('trees', 10)
s.get_hyponyms_plural('weapons', 10)
s.get_parts_plural('houses', 10)
s.get_whole_plural('loaves', 10)

# For verbs
s.get_nouns('purchase', 10)
s.get_nouns_plural('purchase', 10)

# These methods may or may not require Penn Treebank tags.

# Miscellaneous
s.get_most_common_words('VB', 10) # Takes a tag as a parameter
s.get_most_common_tag('dog') # Does not require tag
s.get_words_by_rarity('the boy walked across the wasteland.') # Does not require tag
```

The methods available using the truncated corpus are below. Again, some require tags, and some don't.

```python
import scholar.scholar as sch

# This will load word2vec using the truncated corpus
s = sch.Scholar(slim=True)

# These methods require a Penn Treebank tag

s.get_cosine_similarity('man_NN', 'woman_NN')
s.analogy('king_NN -man_NN woman_NN')
s.exists_in_model('peppercorn_NN')

# The below methods DO NOT require the use of a Penn Treebank tag.

# For singular nouns
s.get_verbs('mailbox', 10) # Where 10 is the number of desired results
s.get_adjectives('mailbox', 10)
s.get_hypernyms('tree', 10)
s.get_hyponyms('weapon', 10)
s.get_parts('house', 10)
s.get_whole('bread', 10)

# These methods may or may not require Penn Treebank tags.

# Miscellaneous
s.get_most_common_words('VB', 10) # Takes a tag as a parameter
s.get_most_common_tag('dog') # This method should never be run with a tag

```
