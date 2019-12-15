import re

prefix_size = 3
suffix_size = 3


class Parser:
	def __init__(self, file, window_size, F2I={}, L2I={}):
		self.file = open(file, 'r')
		self.window_sentences = []
		self.window_sentences_labels = []
		self.window_size = window_size
		self.F2I = F2I
		self.L2I = L2I
		self.prefixes = []
		self.suffixes = []
		self.prefix_F2I = {}
		self.suffix_F2I = {}

	def add_sentence_windows_prefix_suffix(self, sentence_window_list):
		sentence_prefix = list()
		sentence_suffix = list()
		for window in sentence_window_list:
			sentence_prefix.append([word[:prefix_size] for word in window])
			sentence_suffix.append([word[-suffix_size:] for word in window])
		self.prefixes.extend(sentence_prefix)
		self.suffixes.extend(sentence_suffix)

	def create_window_list_from_sentence(self, sentence_list):
		window_sentences = list()
		window_sentences_labels = list()
		if len(sentence_list) < self.window_size:
			raise ValueError("Sentences must be bigger then window size")
		last_element = len(sentence_list) - self.window_size + 1
		for i in range(last_element):
			curr_sentence = [tup[0] for tup in sentence_list[i:i + self.window_size]]
			window_sentences.append(curr_sentence)
			curr_sentence_label = sentence_list[i + self.window_size // 2][1]
			window_sentences_labels.append(curr_sentence_label)
		return window_sentences, window_sentences_labels

	def create_window_test_list_from_sentence(self, sentence_list):
		window_sentences = list()
		if len(sentence_list) < self.window_size:
			raise ValueError("Sentences must be bigger then window size")
		last_element = len(sentence_list) - self.window_size + 1
		for i in range(last_element):
			curr_sentence = [word for word in sentence_list[i:i + self.window_size]]
			window_sentences.append(curr_sentence)
		return window_sentences

	def convert_sentences_to_indexes(self):
		f2i = self.get_f2i()
		l2i = self.get_l2i()
		for sentence in self.window_sentences:
			for index, word in enumerate(sentence):
				if word in f2i:
					sentence[index] = f2i[word]
				else:
					sentence[index] = f2i['']
		for index, label in enumerate(self.window_sentences_labels):
			if label in l2i:
				self.window_sentences_labels[index] = l2i[label]
			else:
				self.window_sentences_labels[index] = l2i['']

	def convert_sub_sentences_to_indexes(self):
		prefix_f2i = self.get_prefix_f2i()
		suffix_f2i = self.get_suffix_f2i()
		for sentence in self.prefixes:
			for index, word in enumerate(sentence):
				if word in prefix_f2i:
					sentence[index] = prefix_f2i[word]
				else:
					sentence[index] = prefix_f2i['']

		for sentence in self.suffixes:
			for index, word in enumerate(sentence):
				if word in suffix_f2i:
					sentence[index] = suffix_f2i[word]
				else:
					sentence[index] = suffix_f2i['']

	def parse_sentences(self, pos=True, convert_digits=True, with_prefix_suffix=False):
		# parse by spaces if post, if ner parse by tab.
		delimiter = ' ' if pos else '\t'
		current_sentence = list()
		for raw in self.file:
			raw_splitted = raw.split('\n')
			raw_splitted = raw_splitted[0].split(delimiter)
			word = raw_splitted[0]
			if word != '':
				# convert all chars to lower case.
				word = word.lower()
				# if we want to convert each digit to be DG for similarity, '300' = '400'.
				if convert_digits:
					word = re.sub('[0-9]', 'DG', word)
				label = raw_splitted[1]
				current_sentence.append((word, label))
			else:
				full_sentence = [('STARTT', 'STARTT'), ('STARTT', 'STARTT')] + current_sentence + [('ENDD', 'ENDD'),
				                                                                                   ('ENDD', 'ENDD')]
				sentences, sentences_labels = self.create_window_list_from_sentence(full_sentence)
				if with_prefix_suffix:
					# if flag is true, create prefix an suffix windows from list of windows.
					self.add_sentence_windows_prefix_suffix(sentences)

				self.window_sentences.extend(sentences)
				self.window_sentences_labels.extend(sentences_labels)
				current_sentence.clear()

		# convert words to indexes
		self.convert_sentences_to_indexes()
		if with_prefix_suffix:
			self.convert_sub_sentences_to_indexes()

	def parse_test_sentences(self, convert_digits=True, with_prefix_suffix=False):
		current_sentence = list()
		for raw in self.file:
			raw_splitted = raw.split('\n')
			word = raw_splitted[0]
			if word != '':
				# convert all chars to lower case.
				word = word.lower()
				# if we want to convert each digit to be DG for similarity, '300' = '400'.
				if convert_digits:
					word = re.sub('[0-9]', 'DG', word)
				current_sentence.append(word)
			else:
				full_sentence = ['STARTT', 'STARTT'] + current_sentence + ['ENDD', 'ENDD']
				sentences = self.create_window_test_list_from_sentence(full_sentence)
				if with_prefix_suffix:
					# if flag is true, create prefix an suffix windows from list of windows.
					self.add_sentence_windows_prefix_suffix(sentences)

				self.window_sentences.extend(sentences)
				current_sentence.clear()

		# convert words to indexes
		self.convert_sentences_to_indexes()
		if with_prefix_suffix:
			self.convert_sub_sentences_to_indexes()

	def get_sentences(self):
		return self.window_sentences

	def get_sentences_prefix(self):
		return self.prefixes

	def get_sentences_suffix(self):
		return self.suffixes

	def get_labels(self):
		return self.window_sentences_labels

	def get_f2i(self):
		if not self.F2I:
			self.F2I = {f: i for i, f in
			            enumerate(list(sorted(set([w for sublist in self.window_sentences for w in sublist]))))}
			self.F2I[''] = len(self.F2I)
		return self.F2I

	def get_prefix_f2i(self):
		if not self.prefix_F2I:
			self.prefix_F2I = {f: i for i, f in
			                   enumerate(list(sorted(set([w for sublist in self.prefixes for w in sublist]))))}
			self.prefix_F2I[''] = len(self.prefix_F2I)
		return self.prefix_F2I

	def get_suffix_f2i(self):
		if not self.suffix_F2I:
			self.suffix_F2I = {f: i for i, f in
			                   enumerate(list(sorted(set([w for sublist in self.suffixes for w in sublist]))))}
			self.suffix_F2I[''] = len(self.suffix_F2I)
		return self.suffix_F2I

	def get_l2i(self):
		if not self.L2I:
			self.L2I = {l: i for i, l in enumerate(list(sorted(set([w for w in self.window_sentences_labels]))))}
			self.L2I[''] = len(self.L2I)
		return self.L2I

	def get_i2l(self):
		i2l = {i: l for l, i in self.L2I.items()}
		i2l[len(i2l)] = ''
		return i2l


if __name__ == '__main__':
	p = Parser("./Dataset/pos/train", window_size=5)
	p.parse_sentences(with_prefix_suffix=True)
