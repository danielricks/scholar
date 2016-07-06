# scholar
This code provides an interface for Word2Vec in Python.

It is intended to be used with [textplayer](https://github.com/kingjamesiv/textplayer).

## Requirements

You will need a Word2Vec .bin file and the file that has the words from Wikipedia along with their word counts.

## Usage

The available methods are demonstrated below. Remember that if you are using a POS-tagged .bin file, you need to append '_NN', etc. onto all your words.

```python
import scholar.scholar as sch
s = sch.Scholar()
print s.get_cosine_similarity('man', 'woman')
print s.analogy('king -man woman')

# For singular nouns
print s.get_verbs('mailbox')
print s.get_adjectives('mailbox')
print s.get_hypernyms('tree')
print s.get_hyponyms('weapon')
print s.get_parts('house')
print s.get_whole('bread')

# For plural nouns
print s.get_verbs_plural('mailboxes')
print s.get_adjectives_plural('mailboxes')
print s.get_hypernyms_plural('trees')
print s.get_hyponyms_plural('weapons')
print s.get_parts_plural('houses')
print s.get_whole_plural('loaves')

# This method takes a Penn Treebank part-of-speech tag and the number of requested results instead of a word.
print s.get_most_common_words('VB', 10)
```
