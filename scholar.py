import word2vec, sys, os, math
import numpy as np
import cPickle as pkl

''' Files used by this class:
		canon_adj.txt		canon_adj_pl.txt
		canon_hypernym.txt	canon_hypernym_pl.txt
		canon_meronym.txt	canon_meronym_pl.txt
		canon_verbs.txt		canon_verbs_pl.txt

		postagged_wikipedia_for_word2vec.bin			(word2vec-compatible file using all pos-tagged Wikipedia, Jan 2016)
		postagged_wikipedia_for_word2vec_30kn3kv.pkl	(scholar-compatible file using top 30k nouns, 3k verbs, same corpus)
		postag_distributions_for_scholar.txt			(pos-tag distributions for all words in Wikipedia)
		postag_distributions_for_scholar_30kn3kv.txt	(pos-tag distributions for top 30k nouns, 3k verbs, same corpus)
'''

class Scholar:

	# Initializes the class
	def __init__(self, slim=False):
		self.slim = slim
		if self.slim:
			self.word2vec_bin_loc = 'scholar/postagged_wikipedia_for_word2vec_30kn3kv.pkl'
			self.tag_distribution_loc = 'scholar/postag_distributions_for_scholar_30kn3kv.txt'
		else:
			self.word2vec_bin_loc = 'scholar/postagged_wikipedia_for_word2vec.bin'
			self.tag_distribution_loc = 'scholar/postag_distributions_for_scholar.txt'
		self.number_of_results = 10
		self.number_analogy_results = 20
		self.autoAddTags = True
		self.load_word2vec(self.word2vec_bin_loc)
		# This is a list of the tags as organized in the text file
		self.tag_list = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
		self.load_tag_counts(self.tag_distribution_loc)

	# Return a list of words from a file
	def load_desired_vocab(self, filename):
		text = open(filename)
		word_list = []
		for line in text:
			word_list.append(line.replace('\n', ''))
		return word_list

	# Loads the word2vec model from a specified file
	def load_word2vec(self, model_filename):
		if self.slim:
		    self.model = pkl.load(open(model_filename, 'r'))
		else:
		    self.model = word2vec.load(model_filename)

	# Loads the part of speech tag counts into a dictionary (words to tag string delimited by '-'s)
	def load_tag_counts(self, tag_count_filename):
		# Read in the tag information for each word from the file
		with open(tag_count_filename) as f:
			word_tag_dist = f.read()

		# Save each word to a list of tags in a global dictionary
		self.word_to_tags = {}
		for line in word_tag_dist.split():
			pieces = line.split('.')
			word = pieces[0]
			tags = pieces[1].split('-')
			tags = map(int, tags)
			self.word_to_tags[word] = tags

	# Return the cosine similarity of vectors for two specified words
	def get_cosine_similarity(self, word1, word2):
		vec1 = self.model.get_vector(word1)
		vec2 = self.model.get_vector(word2)
		dividend = np.dot(vec1, vec2)
		divisor = np.linalg.norm(vec1) * np.linalg.norm(vec2)
		result = dividend / divisor
		return result

	# Return the angle between two vectors (assumes a hypersphere)
	def get_angle(self, word1, word2):
		vec1 = self.model.get_vector(word1)
		vec2 = self.model.get_vector(word2)
		unit_vec1 = vec1 / np.linalg.norm(vec1)
		unit_vec2 = vec2 / np.linalg.norm(vec2)
		return np.arccos(np.clip(np.dot(unit_vec1, unit_vec2), -1.0, 1.0))

	# Return the analogy results for a list of words (input: "king -man woman")
	def analogy(self, words_string):
		positives, negatives = self.get_positives_and_negatives(words_string.split())
		return self.get_results_for_words(positives, negatives)

	# Takes a list of words (ie 'king woman -man') and separates them into two lists (ie '["king", "woman"], ["man"]')
	def get_positives_and_negatives(self, words):
		positives = []
		negatives = []
		for x in range(len(words)):
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
		for word_index in range( len(sentence) ):
			total_vector = np.array([0.0] * 100)
			# Add up the vectors for every other word in the sentence...
			for vector_index in range( len(word_vectors) ): 
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
	def get_verbs(self, noun, number_of_user_results):
		return self.get_canonical_results_for_nouns(noun, 'VB', 'scholar/canon_verbs.txt', False, number_of_user_results)

	# Returns the canonical results for nouns
	def get_nouns(self, verb, number_of_user_results):
		return self.get_canonical_results_for_verbs(verb, 'scholar/canon_verbs.txt', False, number_of_user_results)

	# Returns the canonical results for adjectives
	def get_adjectives(self, noun, number_of_user_results):
		return self.get_canonical_results_for_nouns(noun, 'JJ', 'scholar/canon_adj.txt', False, number_of_user_results)

	# Returns the canonical results for hypernyms (generalized words)
	def get_hypernyms(self, noun, number_of_user_results):
		return self.get_canonical_results_for_nouns(noun, 'HYPER', 'scholar/canon_hypernym.txt', False, number_of_user_results)

	# Returns the canonical results for hyponyms (specific words)
	def get_hyponyms(self, noun, number_of_user_results):
		return self.get_canonical_results_for_nouns(noun, 'HYPO', 'scholar/canon_hypernym.txt', False, number_of_user_results)

	# Returns the canonical results for parts of the given noun
	def get_parts(self, noun, number_of_user_results):
		return self.get_canonical_results_for_nouns(noun, 'PARTS', 'scholar/canon_meronym.txt', False, number_of_user_results)

	# Returns the canonical results for things the noun could be a part of
	def get_whole(self, noun, number_of_user_results):
		return self.get_canonical_results_for_nouns(noun, 'WHOLE', 'scholar/canon_meronym.txt', False, number_of_user_results)

	# Returns the canonical results for verbs (plural)
	def get_verbs_plural(self, noun, number_of_user_results):
		return self.get_canonical_results_for_nouns(noun, 'VB', 'scholar/canon_verbs_pl.txt', True, number_of_user_results)

	# Returns the canonical results for nouns (plural)
	def get_nouns_plural(self, verb, number_of_user_results):
		return self.get_canonical_results_for_verbs(verb, 'scholar/canon_verbs.txt', True, number_of_user_results)

	# Returns the canonical results for adjectives (plural)
	def get_adjectives_plural(self, noun, number_of_user_results):
		return self.get_canonical_results_for_nouns(noun, 'JJ', 'scholar/canon_adj_pl.txt', True, number_of_user_results)

	# Returns the canonical results for hypernyms (generalized words) (plural)
	def get_hypernyms_plural(self, noun, number_of_user_results):
		return self.get_canonical_results_for_nouns(noun, 'HYPER', 'scholar/canon_hypernym_pl.txt', True, number_of_user_results)

	# Returns the canonical results for hyponyms (specific words) (plural)
	def get_hyponyms_plural(self, noun, number_of_user_results):
		return self.get_canonical_results_for_nouns(noun, 'HYPO', 'scholar/canon_hypernym_pl.txt', True, number_of_user_results)

	# Returns the canonical results for parts of the given noun (plural)
	def get_parts_plural(self, noun, number_of_user_results):
		return self.get_canonical_results_for_nouns(noun, 'PARTS', 'scholar/canon_meronym_pl.txt', True, number_of_user_results)

	# Returns the canonical results for things the noun could be a part of (plural)
	def get_whole_plural(self, noun, number_of_user_results):
		return self.get_canonical_results_for_nouns(noun, 'WHOLE', 'scholar/canon_meronym_pl.txt', True, number_of_user_results)

	# Returns canonical results for specified relationships between words
	# As an aside, this is simply returning the results of all the analogies from all the canonical pairs.
	# Occasionally it returns unexpected tags (ie user requested a list of adjectives related to a noun, 
	# and got mostly adjectives but also one preposition). Be aware of this if it matters.
	def get_canonical_results_for_nouns(self, noun, query_tag, canonical_tag_filename, plural, number_of_user_results):
		if self.autoAddTags:
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
				if result in result_map.keys():
					result_map[result] += 1
				else:
					result_map[result] = 1
		final_results = []
		current_max = number_of_user_results
		# While we haven't reached the requested number of results and the number of possible matches is within reason...
		while len(final_results) < number_of_user_results and current_max > 0:
			# ...for every key in the results...
			for key in result_map.keys():
				# ...if the number of times a result has been seen equals the current 'number of matches'...
				if result_map[key] == current_max:
					# ...add it to the list. (This is so that the results are sorted to the list in order of popularity)
					final_results.append(key)
			current_max -= 1
		if len(final_results) >= number_of_user_results:
			return final_results[0:number_of_user_results]
		return final_results

	# Returns canonical results for specified relationships between words
	# As an aside, this is simply returning the results of all the analogies from all the canonical pairs.
	# Occasionally it returns unexpected tags (ie user requested a list of adjectives related to a noun, 
	# and got mostly adjectives but also one preposition). Be aware of this if it matters.
	def get_canonical_results_for_verbs(self, verb, canonical_tag_filename, plural, number_of_user_results):
		canonical_pairs = open(canonical_tag_filename)
		result_map = {}
		# For every line in the file of canonical pairs...
		for line in canonical_pairs:
			# ...split into separate words...
			words = line.split()
			if plural:
				query_string = words[1] + '_NNS' + ' -' + words[0] + '_VB ' + verb + '_VB'
			else:
				query_string = words[1] + '_NN' + ' -' + words[0] + '_VB ' + verb + '_VB'

			# ...performs an analogy using the words...
			try:
				result_list = self.analogy(query_string)
			except:
				result_list = []
			# ...and adds those results to a map (sorting depending on popularity, Poll method)
			for result in result_list:
				if result in result_map.keys():
					result_map[result] += 1
				else:
					result_map[result] = 1
		final_results = []
		current_max = number_of_user_results
		# While we haven't reached the requested number of results and the number of possible matches is within reason...
		while len(final_results) < number_of_user_results and current_max > 0:
			# ...for every key in the results...
			for key in result_map.keys():
				# ...if the number of times a result has been seen equals the current 'number of matches'...
				if result_map[key] == current_max:
					# ...add it to the list. (This is so that the results are sorted to the list in order of popularity)
					final_results.append(key)
			current_max -= 1
		if len(final_results) >= number_of_user_results:
			return final_results[0:number_of_user_results]
		return final_results

	def get_most_common_words(self, pos_tag, number_of_results):
		# If the tag doesn't exist, return nothing
		if pos_tag not in self.tag_list or not os.path.exists(self.tag_distribution_loc):
			return []

		# Get the index of the specific tag requested in the list above
		tag_index = self.tag_list.index(pos_tag)

		# Read in the tag information for each word from the file
		with open(self.tag_distribution_loc) as f:
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
		if (sys.version_info > (3, 0)):
			taglist = sorted(taglist, key=lambda k: int(k))
		else:
			taglist.sort()
		# Reverse the list (to highest to lowest)
		taglist.reverse()
		# Add the words for each tag number to a list
		for tag in taglist:
			common_words += tag_to_word[tag]

		# Only return the number of results specified by the user
		return common_words[:number_of_results]

	def get_words_by_rarity(self, sentence):
		# Clean up input sentence (remove punctuation and unnecessary white space)
		sentence = sentence.replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ').replace(':', ' ').replace(';', ' ').replace('-', ' ')
		while '  ' in sentence:
			sentence = sentence.replace('  ', ' ')
		# Create dictionary of words to their popularities
		word_to_pop = {}
		for word in sentence.split():
			word_to_pop[word] = self.get_word_popularity(word)
		# Return list of words sorted by popularities
		return sorted(word_to_pop, key=word_to_pop.__getitem__)

	# Returns the popularity of a word (without a tag)
	def get_word_popularity(self,word):
		try:
			popularity = 0
			for tag_amount in self.word_to_tags[word]:
				popularity += int(tag_amount)#int(self.word_to_tags[word][tag_amount])
			return popularity
		except:
			if (sys.version_info > (3, 0)):
				return math.inf
			else:
				return float('inf')

	# Returns the most common tag for a specific word
	def get_most_common_tag(self, word):
		word_tags = self.word_to_tags[word]
		return self.tag_list[word_tags.index(max(word_tags))]

	# Returns True if the word/tag pair exists in the Wikipedia corpus
	def exists_in_model(self, word):
		try:
			vector = self.model.get_vector(word)
			return True
		except:
			return False

