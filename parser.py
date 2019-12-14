class Parser:
	def __init__(self, file, window_size):
		self.file = open(file, 'r')
		self.window_sentences = []
		self.window_sentences_labels = []
		self.window_size = window_size
		self.F2I = {}
		self.L2I = {}

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

	def convert_sentences_to_indexes(self):
		f2i = self.get_f2i()
		l2i = self.get_l2i()
		for sentence in self.window_sentences:
			for index, word in enumerate(sentence):
				sentence[index] = f2i[word]

		self.window_sentences_labels = [l2i[w] for w in self.window_sentences_labels]

	def parse_sentences(self, f_vocab={}, l_vocab={}):
		current_sentence = list()
		for raw in self.file:
			raw_splitted = raw.split('\n')
			raw_splitted = raw_splitted[0].split(' ')
			word = raw_splitted[0]
			if word != '':
				label = raw_splitted[1]
				current_sentence.append((word, label))
				# if given a vocab, check if the word exist in it. if not replace it with ('','')
				if f_vocab and l_vocab and (word not in f_vocab or label not in l_vocab):
					word, label = '', ''
				current_sentence.append((word, label))
			else:
				full_sentence = [('STARTT', 'STARTT'), ('STARTT', 'STARTT')] + current_sentence + [('ENDD', 'ENDD'),
				('ENDD', 'ENDD')]
				sentences, sentences_labels = self.create_window_list_from_sentence(full_sentence)
				self.window_sentences.extend(sentences)
				self.window_sentences_labels.extend(sentences_labels)
				current_sentence.clear()

		# convert words to indexes
		self.convert_sentences_to_indexes()

	def get_sentences(self):
		return self.window_sentences

	def get_labels(self):
		return self.window_sentences_labels

	def get_f2i(self):
		if not self.F2I:
			self.F2I = {f: i for i, f in enumerate(list(sorted(set([w for sublist in self.window_sentences for w in sublist]))))}
			self.F2I[''] = len(self.F2I)
		return self.F2I

	def get_l2i(self):
		if not self.L2I:
			self.L2I = {l: i for i, l in enumerate(list(sorted(set([w for w in self.window_sentences_labels]))))}
			self.L2I[''] = len(self.L2I)
		return self.L2I


if __name__ == '__main__':
	p = Parser("./Dataset/pos/train", window_size=5)
	p.parse_sentences()
