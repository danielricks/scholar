import word2vec, sys, os
import numpy as np

''' Files used by this class:
		canon_adj.txt
		canon_hypernym.txt
		canon_meronym.txt
		canon_verbs.txt
		des_words_wiki_parsey_100.txt
		wikipedia_articles_parsey.bin

Class Summary: Scholar()
Methods:	get_verbs([singular noun])
 			get_adjectives([singular noun])
			get_hypernyms([singular noun])
			get_hyponyms([singular noun])
			get_parts([singular noun])
			get_whole([singular noun])
 			get_cosine_similarity([singular noun], [singular noun])
			get_analogy([string consisting of words separated by spaces, with '-' preceding at least one word])
'''

class Scholar:

	# Initializes the class
	def __init__(self):
		self.number_of_results = 10
		self.number_analogy_results = 20
#		desired_vocab = self.load_desired_vocab('scholar/des_words_wiki_parsey_100.txt')
		self.load_word2vec('scholar/wikipedia_articles_parsey.bin')

	# Return a list of words from a file
	def load_desired_vocab(self, filename):
		text = open(filename)
		word_list = []
		for line in text:
			word_list.append(line.replace('\n', ''))
		return word_list

	# Loads the word2vec model from a specified file
	def load_word2vec(self, model_filename):
		self.model = word2vec.load(model_filename)

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

	# Return the analogy results for a list of words (input: "king -man woman")
	def analogy_2(self, words_string):
		positives, negatives = self.get_positives_and_negatives(words_string.split())
		return self.get_results_for_words_2(positives, negatives)

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
		indexes, metrics = self.model.analogy(pos=positives, neg=negatives, n=self.number_analogy_results)
		results = self.model.generate_response(indexes, metrics).tolist()
		return self.format_output(results)

	# Returns the results of entering a list of positive and negative words into word2vec
	def get_results_for_words_2(self, positives, negatives):
		indexes, metrics = self.model.analogy(pos=positives, neg=negatives, n=self.number_analogy_results)
		results = self.model.generate_response(indexes, metrics).tolist()
		word_tags = []
		for word_value in results:
			new_tuple = ( str(word_value[0]), word_value[1] )
			word_tags.append(new_tuple)
		return word_tags

	# Changes the output from a list of tuples (u'man', 0.816015154188), ... to a list of single words
	def format_output(self, output):
		words = []
		for word_value in output:
			words.append(str(word_value[0]))
		return words

	# Returns the canonical results for verbs
	def get_verbs(self, noun):
		return self.get_canonical_results(noun, 'VB', 'scholar/canon_verbs.txt')

	# Returns the canonical results for adjectives
	def get_adjectives(self, noun):
		return self.get_canonical_results(noun, 'JJ', 'scholar/canon_adj.txt')

	# Returns the canonical results for hypernyms (generalized words)
	def get_hypernyms(self, noun):
		return self.get_canonical_results(noun, 'HYPER', 'scholar/canon_hypernym.txt')

	# Returns the canonical results for hyponyms (specific words)
	def get_hyponyms(self, noun):
		return self.get_canonical_results(noun, 'HYPO', 'scholar/canon_hypernym.txt')

	# Returns the canonical results for parts of the given noun
	def get_parts(self, noun):
		return self.get_canonical_results(noun, 'PARTS', 'scholar/canon_meronym.txt')

	# Returns the canonical results for things the noun could be a part of
	def get_whole(self, noun):
		return self.get_canonical_results(noun, 'WHOLE', 'scholar/canon_meronym.txt')

	# Returns canonical results for specified relationships between words
	def get_canonical_results(self, noun, query_tag, canonical_tag_filename):
		canonical_pairs = open(canonical_tag_filename)
		result_map = {}
		# For every line in the file of canonical pairs...
		for line in canonical_pairs:
			# ...split into separate words...
			words = line.split()
			if query_tag == 'VB' or query_tag == 'JJ':
				query_string = words[0] + '_' + query_tag + ' -' + words[1] + '_NN ' + noun
			elif query_tag == 'HYPER':
				query_string = words[0] + '_NN -' + words[1] + '_NN ' + noun
			elif query_tag == 'HYPO':
				query_string = words[1] + '_NN -' + words[0] + '_NN ' + noun
			elif query_tag == 'PARTS':
				query_string = '-' + words[0] + '_NN ' + words[1] + '_NN ' + noun
			elif query_tag == 'WHOLE':
				query_string = '-' + words[1] + '_NN ' + words[0] + '_NN ' + noun
			# ...performs an analogy using the words...
			try:
				result_list = self.analogy(query_string)
			except:
				result_list = []
			# ...and adds those results to a map (sorting depending on popularity, Poll method)
			for result in result_list:
				if result_map.has_key(result):
					result_map[result] += 1
				else:
					result_map[result] = 1
		final_results = []
		current_max = self.number_of_results
		# While we haven't reached the requested number of results and the number of possible matches is within reason...
		while len(final_results) < self.number_of_results and current_max > 0:
			# ...for every key in the results...
			for key in result_map.keys():
				# ...if the number of times a result has been seen equals the current 'number of matches'...
				if result_map[key] == current_max:
					# ...add it to the list. (This is so that the results are sorted to the list in order of popularity)
					final_results.append(key)
			current_max -= 1
		return final_results


	# Separate out from verb and adjective options, provide filename for canonical pairs
	def get_related_words_2(self, noun, query_tag):
		canonical_tag_filename = ''
		if query_tag == 'VB':
			canonical_tag_filename = 'scholar/canon_verbs.txt'
		elif query_tag == 'JJ':
			canonical_tag_filename = 'scholar/canon_adj.txt'
		if canonical_tag_filename == '':
			return
		return self.get_words_2(noun, query_tag, canonical_tag_filename)

	# Highest Score method
	# Returns a list of likely verbs for a given noun
	def get_words_2(self, noun, query_tag, canonical_tag_filename):
		canonical_pairs = open(canonical_tag_filename)
		score_to_verb = {}
		# Run analogy on the word versus canonical pairs
		for line in canonical_pairs:
			words = line.split()
			query_string = words[0] + '_' + query_tag + ' -' + words[1] + '_NN ' + noun
			verb_score_list = self.analogy_2(query_string)
			# For every verb returned...
			for verb_score in verb_score_list:
				score_to_verb[verb_score[1]] = verb_score[0]
#		print score_to_verb
		new_score_list = score_to_verb.keys()
		new_score_list.sort()
		new_score_list.reverse()
		final_results = []
		current_index = 0
		while len(final_results) < self.number_of_results and current_index < self.number_of_results:
			word = score_to_verb[new_score_list[current_index]]
			if word not in final_results:
				final_results.append(word)
			current_index += 1
		return final_results

	def get_antonyms(self, noun):
		pass

