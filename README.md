# scholar
This code provides an interface for Word2Vec in Python.

It is intended to be used with [textplayer](https://github.com/kingjamesiv/textplayer).

## Requirements

You will need a Word2Vec .bin file.

## Usage

There are currently only a couple methods available. They are demonstrated below.

'''python
s = Scholar()
print s.get_verbs('mailbox')
print s.get_cosine_similarity('man', 'woman')
print s.analogy('king -man woman')
'''

