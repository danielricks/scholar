# scholar
This code provides an interface for Word2Vec in Python. It can use a part-of-speech tagged corpus to query for specific parts of speech. It was intended to be used with [autoplay](https://github.com/danielricks/autoplay), a learning environment for interactive fiction.

Using linear algebra, we were able to pull affordances (relevant verbs) for nouns out of word2vec, with varying success on other parts of speech. This interface provides methods that perform those operations.

## Requirements

Our processed files are available [here](https://drive.google.com/open?id=1srOUFidQ9fV240wyF7GW4eqF6raCawBV).

Provided are several bin files: (1) an untagged copy of Wikipedia from January 2016, (2) a part-of-speech tagged copy of Wikipedia, and (3) a truncated version of 2. Per standard natural language processing, we used [Penn Treebank tags](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html) to denote parts of speech. We tagged our copy of Wikipedia (using [Parsey McParseface](https://github.com/tensorflow/models/tree/master/syntaxnet)) and used that as input to word2vec.

The truncated corpus contains the top 30,000 most popular singular nouns on Wikipedia, and the top 3,000 verbs. It is stored in pickle format and loads much faster than the full binary, but also loses access to some methods. I would honestly recommend using the truncated corpus if you can, since the speed-up is significant, and most of the real functionality is included. The above download link also provides pre-computed distributions of word-to-pos tags, to speed up computation during runtime.

## Usage

There are 3 options for using scholar.
1. Untagged words (gives basic word2vec functionality, 4 million words)
2. pos-tagged words (gives basic word2vec functionality and part-of-speech queries, 4 million words)
3. Slim pos-tagged words (limited word2vec functionality and part-of-speech queries, 33k most popular nouns and verbs)

### Usage Examples

1. The available methods using the full untagged corpus are demonstrated below. Most methods don't require tags, but some miscellaneous methods that do require tags are maintained for convenience.

```python
import scholar.scholar as sch

# This will load word2vec using the full untagged corpus
s = sch.Scholar(tags=False)

# These methods require a Penn Treebank tag

s.get_cosine_similarity('man', 'woman')
s.analogy('king -man woman')
s.exists_in_model('peppercorn')
s.exists_in_model_untagged('dog') # Using an untagged corpus, these methods are identical
s.get_angle('dog', 'cat')
dog_vec = s.get_vector('dog')
s.get_words(dog_vec, 10)

# These methods may or may not require Penn Treebank tags.

# Miscellaneous
s.get_most_common_words('VB', 10) # Takes a tag as a parameter
s.get_most_common_tag('dog')
s.get_words_by_rarity('the boy walked across the wasteland.')
s.exists_in_model('dog')
```

2. The available methods using the full pos-tagged corpus are demonstrated below. Some methods require a pos-tag, and some of them don't.

```python
import scholar.scholar as sch

# This will load word2vec using the full tagged corpus
s = sch.Scholar()

# These methods require a Penn Treebank tag

s.get_cosine_similarity('man_NN', 'woman_NN')
s.analogy('king_NN -man_NN woman_NN')
s.exists_in_model('peppercorn_NN')
s.get_angle('dog_NN', 'cat_NN')
dog_vec = s.get_vector('dog_NN')
s.get_words(dog_vec, 10)

# The below methods DO NOT require the use of a Penn Treebank tag, but will accept them.

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
s.exists_in_model('dog_NN') # Requires tag
s.exists_in_model_untagged('dog') # Does not require tag
```

3. The methods available using the slim pos-tagged corpus are below. Again, some require tags, and some don't.

```python
import scholar.scholar as sch

# This will load word2vec using the truncated corpus
s = sch.Scholar(slim=True)

# These methods require a Penn Treebank tag

s.get_cosine_similarity('man_NN', 'woman_NN')
s.analogy('king_NN -man_NN woman_NN')
s.exists_in_model('peppercorn_NN')
s.get_angle('dog_NN', 'cat_NN')
dog_vec = s.get_vector('dog_NN')
s.get_words(dog_vec, 10)

# The below methods DO NOT require the use of a Penn Treebank tag, but will accept them.

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
s.exists_in_model('dog_NN') # Requires tag
s.exists_in_model_untagged('dog') # Does not require tag
```
