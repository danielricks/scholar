import word2vec, sys, os, math
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
		self.autoAddNounTags = False
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

	# Changes the output from a list of tuples (u'man', 0.816015154188), ... to a list of single words
	def format_output(self, output):
		words = []
		for word_value in output:
			words.append(str(word_value[0]))
		return words

	# Returns a list of the words in a tagged sentence ordered by salience (as determined by Word2Vec)
	def get_words_by_salience(self, sentence):
		sentence = sentence.split()
		word_vectors = []
		# Get the vectors for every word in the sentence
		for tagged_word in sentence:
			word_vectors.append(self.model[tagged_word])
		word_salience = {}
		# For every word in the sentence...
		for word_index in xrange( len(sentence) ):
			total_vector = np.array([0.0] * 100)
			# Add up the vectors for every other word in the sentence...
			for vector_index in xrange( len(word_vectors) ): 
				if word_index != vector_index:
					total_vector += word_vectors[vector_index]
			# Find the average for those vectors
			average_vector = total_vector / float( len(word_vectors) - 1 )
			# Take the difference of the average vector and the current word vector
			difference_list = ( average_vector - word_vectors[word_index] ).tolist()
			difference_scalar = 0
			# For every scalar in the difference vector...
			for difference_number in difference_list:
				# Add that squared number to a single scalar
				difference_scalar += math.pow(difference_number, 2)
			# The square root of that single scalar is the key in a dictionary
			word_salience[ math.sqrt(difference_scalar) ] = sentence[word_index]
		words_sorted_by_salience = []
		# Add words in order of lowest salience to highest
		for key in sorted(word_salience.iterkeys()):
			words_sorted_by_salience.append(word_salience[key])
		# Reverse the list
		words_sorted_by_salience.reverse()
		return words_sorted_by_salience

	# Returns the canonical results for verbs
	def get_verbs(self, noun):
		return self.get_canonical_results(noun, 'VB', 'scholar/canon_verbs.txt', False)

	# Returns the canonical results for adjectives
	def get_adjectives(self, noun):
		return self.get_canonical_results(noun, 'JJ', 'scholar/canon_adj.txt', False)

	# Returns the canonical results for hypernyms (generalized words)
	def get_hypernyms(self, noun):
		return self.get_canonical_results(noun, 'HYPER', 'scholar/canon_hypernym.txt', False)

	# Returns the canonical results for hyponyms (specific words)
	def get_hyponyms(self, noun):
		return self.get_canonical_results(noun, 'HYPO', 'scholar/canon_hypernym.txt', False)

	# Returns the canonical results for parts of the given noun
	def get_parts(self, noun):
		return self.get_canonical_results(noun, 'PARTS', 'scholar/canon_meronym.txt', False)

	# Returns the canonical results for things the noun could be a part of
	def get_whole(self, noun):
		return self.get_canonical_results(noun, 'WHOLE', 'scholar/canon_meronym.txt', False)

	# Returns the canonical results for verbs (plural)
	def get_verbs_plural(self, noun):
		return self.get_canonical_results(noun, 'VB', 'scholar/canon_verbs_pl.txt', True)

	# Returns the canonical results for adjectives (plural)
	def get_adjectives_plural(self, noun):
		return self.get_canonical_results(noun, 'JJ', 'scholar/canon_adj_pl.txt', True)

	# Returns the canonical results for hypernyms (generalized words) (plural)
	def get_hypernyms_plural(self, noun):
		return self.get_canonical_results(noun, 'HYPER', 'scholar/canon_hypernym_pl.txt', True)

	# Returns the canonical results for hyponyms (specific words) (plural)
	def get_hyponyms_plural(self, noun):
		return self.get_canonical_results(noun, 'HYPO', 'scholar/canon_hypernym_pl.txt', True)

	# Returns the canonical results for parts of the given noun (plural)
	def get_parts_plural(self, noun):
		return self.get_canonical_results(noun, 'PARTS', 'scholar/canon_meronym_pl.txt', True)

	# Returns the canonical results for things the noun could be a part of (plural)
	def get_whole_plural(self, noun):
		return self.get_canonical_results(noun, 'WHOLE', 'scholar/canon_meronym_pl.txt', True)

	# Returns canonical results for specified relationships between words
	def get_canonical_results(self, noun, query_tag, canonical_tag_filename, plural):
		if self.autoAddNounTags:
			noun += '_NNS' if plural else '_NN'
		canonical_pairs = open(canonical_tag_filename)
		result_map = {}
		# For every line in the file of canonical pairs...
		for line in canonical_pairs:
			# ...split into separate words...
			words = line.split()
			if plural:
				if query_tag == 'VB' or query_tag == 'JJ':
					query_string = words[0] + '_' + query_tag + ' -' + words[1] + '_NNS ' + noun
				elif query_tag == 'HYPER':
					query_string = words[0] + '_NNS -' + words[1] + '_NNS ' + noun
				elif query_tag == 'HYPO':
					query_string = words[1] + '_NNS -' + words[0] + '_NNS ' + noun
				elif query_tag == 'PARTS':
					query_string = '-' + words[0] + '_NNS ' + words[1] + '_NNS ' + noun
				elif query_tag == 'WHOLE':
					query_string = '-' + words[1] + '_NNS ' + words[0] + '_NNS ' + noun
			else:
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
		if len(final_results) > self.number_analogy_results:
			return final_results[0:self.number_analogy_results]
		return final_results

	def get_most_common_words(self, pos_tag, number_of_results):
		# This is a list of the tags as organized in the text file
		tag_list = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

		# If the tag doesn't exist, return nothing
		if pos_tag not in tag_list or not os.path.exists('scholar/enwiki_dist_parsey.txt'):
			return []

		# Get the index of the specific tag requested in the list above
		tag_index = tag_list.index(pos_tag)

		# Read in the tag information for each word from the file
		with open('scholar/enwiki_dist_parsey.txt') as f:
			word_tag_dist = f.read()

		tag_to_word = {}

		# For each of the lines in the text file... (dog.0-0-0-0-0-4-0-0-90-3-0-0-etc.)
		for line in word_tag_dist.split():
			pieces = line.split('.')
			word = pieces[0]
			tags = pieces[1].split('-')
			current_tag = int(tags[tag_index])
			# Add to the dictionary of tag numbers to words
			try:
				tag_to_word[current_tag].append(word)
			except:
				tag_to_word[current_tag] = []
				tag_to_word[current_tag].append(word)

		common_words = []
		taglist = tag_to_word.keys()
		# Sort the list of tag numbers from lowest to highest
		taglist.sort()
		# Reverse the list (to highest to lowest)
		taglist.reverse()
		# Add the words for each tag number to a list
		for tag in taglist:
			common_words += tag_to_word[tag]

		# Only return the number of results specified by the user
		return common_words[:number_of_results]

#----------------------Highest Score Method----------------------

	# Return the analogy results for a list of words (input: "king -man woman")
	def analogy_2(self, words_string):
		positives, negatives = self.get_positives_and_negatives(words_string.split())
		return self.get_results_for_words_2(positives, negatives)

	# Returns the results of entering a list of positive and negative words into word2vec
	def get_results_for_words_2(self, positives, negatives):
		indexes, metrics = self.model.analogy(pos=positives, neg=negatives, n=self.number_analogy_results)
		results = self.model.generate_response(indexes, metrics).tolist()
		word_tags = []
		for word_value in results:
			new_tuple = ( str(word_value[0]), word_value[1] )
			word_tags.append(new_tuple)
		return word_tags

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
		return None

