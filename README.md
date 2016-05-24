# scholar
This code provides an interface for Word2Vec in Python.

It is intended to be used with [textplayer](https://github.com/kingjamesiv/textplayer).

## Requirements

You will need a Word2Vec .bin file. Remember that if you are using a POS-tagged .bin file, you need to append '_NN', etc. onto all your words.

## Usage

There are currently only a couple methods available. They are demonstrated below.

```python
import scholar.scholar as sch
s = sch.Scholar()
print s.get_verbs('mailbox')
print s.get_adjectives('mailbox')
print s.get_hypernyms('tree')
print s.get_hyponyms('weapon')
print s.get_parts('house')
print s.get_whole('bread')
print s.get_cosine_similarity('man', 'woman')
print s.analogy('king -man woman')
```
