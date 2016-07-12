# scholar
This code provides an interface for Word2Vec in Python.

It is intended to be used with [autoplay](https://github.com/danielricks/autoplay).

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
print s.get_verbs('mailbox', 10) # Where 10 is the number of desired results
print s.get_adjectives('mailbox', 10)
print s.get_hypernyms('tree', 10)
print s.get_hyponyms('weapon', 10)
print s.get_parts('house', 10)
print s.get_whole('bread', 10)

# For plural nouns
print s.get_verbs_plural('mailboxes', 10)
print s.get_adjectives_plural('mailboxes', 10)
print s.get_hypernyms_plural('trees', 10)
print s.get_hyponyms_plural('weapons', 10)
print s.get_parts_plural('houses', 10)
print s.get_whole_plural('loaves', 10)

# For verbs
print s.get_nouns('purchase', 10)
print s.get_nouns_plural('purchase', 10)

# Penn Treebank methods
print s.get_most_common_words('VB', 10)
print s.get_most_common_tag('house')
```
