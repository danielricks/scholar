import word2vec, sys
import numpy as np

''' This uses des_words.txt and canon.txt
'''

class Scholar:

	# Initializes the class
	def __init__(self):
		self.number_of_results = 20
		desired_vocab = self.load_desired_vocab('des_words_wiki1000.txt')
		self.canonical_pairs_filename = 'canon.txt'
		self.load_word2vec('wikipedia_articles_tagged.bin', desired_vocab)

	# Return a list of words from a file
	def load_desired_vocab(self, filename):
		text = open(filename)
		word_list = []
		for line in text:
			word_list.append(line.replace('\n', ''))
		return word_list

	# Loads the word2vec model from a specified file
	def load_word2vec(self, model_filename, des_vocab):
		self.model = word2vec.load(model_filename)#, desired_vocab=des_vocab)

	# Return the cosine similarity of vectors for two specified words
	def get_cosine_similarity(self, word1, word2):
		vec1 = self.model.get_vector(word1)
		vec2 = self.model.get_vector(word2)
		dividend = np.dot(vec1, vec2)
		divisor = np.linalg.norm(vec1) * np.linalg.norm(vec2)
		result = dividend / divisor
		return result

	# Return the analogy results for a list of words (input: "king -man woman")
	def analogy(self, words_string):
		positives, negatives = self.get_positives_and_negatives(words_string.split())
		return self.get_results_for_words(positives, negatives)

	# Takes a list of words (ie 'king woman -man') and separates them into two lists (ie '["king", "woman"], ["man"]')
	def get_positives_and_negatives(self, words):
		positives = []
		negatives = []
		for x in xrange(len(words)):
			word_arg = words[x]
			if word_arg.startswith('-'):
				negatives.append(word_arg[1:])
			else:
				positives.append(word_arg)
		return positives, negatives

	# Returns the results of entering a list of positive and negative words into word2vec
	def get_results_for_words(self, positives, negatives):
		indexes, metrics = self.model.analogy(pos=positives, neg=negatives, n=self.number_of_results)
		results = self.model.generate_response(indexes, metrics).tolist()
		return self.format_output(results)

	# Changes the output from a list of tuples (u'man', 0.816015154188), ... to a list of single words
	def format_output(self, output):
		words = []
		for word_value in output:
			words.append(str(word_value[0]))
		return words

	# Returns a list of likely verbs for a given noun
	def get_verbs(self, noun):
		canonical_pairs = open(self.canonical_pairs_filename)
		verb_map = {}
		# Run analogy on the word versus canonical pairs
		for line in canonical_pairs:
			words = line.split()
			query_string = words[0] + '_VB -' + words[1] + '_NN ' + noun + '_NN'
			verb_list = self.analogy(query_string)
			# For every verb returned...
			for verb in verb_list:
				# ...if that verb already exists, increase the count...
				if verb_map.has_key(verb):
					verb_map[verb] += 1
				# ...else set it to one.
				else:
					verb_map[verb] = 1
		final_results = []
		current_max = self.number_of_results
		# While the length of the result list than some arbitrary amount...
		while len(final_results) < self.number_of_results:
			print current_max
			for key in verb_map.keys():
				# ...if the verb has count equal to the current max (the current highest possible value)...
				if verb_map[key] == current_max:
					# ...add it to the list.
					print key
					final_results.append(key)
			current_max -= 1
		return final_results
'''
ze_word = sys.argv[1]

s = Scholar()
print s.get_verbs(ze_word)'''
#print s.get_cosine_similarity('man', 'woman')
#print s.analogy('king -man woman')

