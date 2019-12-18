import re

START = "*START*"
END = "*END*"
UNKNOWN = '*UNKNOWN*'


def get_file_directory(data_name, data_kind):
	return "./data/{0}/{1}".format(data_name, data_kind)


class Parser:
	def __init__(self, window_size, data_name='pos', data_kind="train", F2I={}, L2I={}):
		file_dir = get_file_directory(data_name, data_kind)
		self.file = open(file_dir, 'r')
		self.window_sentences = []
		self.window_sentences_labels = []
		self.window_size = window_size
		self.sentences_words = []
		self.sentences_labels = []
		self.F2I = F2I
		self.L2I = L2I
		self.is_pos = (data_name == 'pos')
		self.data_kind = data_kind

	def parse_to_indexed_windows(self, convert_digits=True, to_lower=True):
		self.parse_sentences(convert_digits=True, to_lower=True)
		self.create_windows_list_from_sentences()
		self.convert_sentences_windows_to_indexes()

	def create_windows_list_from_sentences(self):
		for sentence in self.sentences_words:
			if len(sentence) < self.window_size:
				raise ValueError("Sentences must be bigger then window size")
			current_sentence_window = list()
			# curr_sentence_label = ""
			last_element = len(sentence) - self.window_size + 1
			for i in range(last_element):
				curr_sentence = [word for word in sentence[i:i + self.window_size]]
				current_sentence_window.append(curr_sentence)
			self.window_sentences.extend(current_sentence_window)

		if self.data_kind != "test":
			for sentence_labels in self.sentences_labels:
				last_element = len(sentence_labels) - self.window_size + 1
				for i in range(last_element):
					curr_sentence_label = sentence_labels[i + self.window_size // 2]
					self.window_sentences_labels.append(curr_sentence_label)

	def convert_sentences_windows_to_indexes(self):
		f2i = self.get_f2i()
		l2i = self.get_l2i()
		for sentence in self.window_sentences:
			for index, word in enumerate(sentence):
				if word in f2i:
					sentence[index] = f2i[word]
				else:
					sentence[index] = f2i[UNKNOWN]
		if self.data_kind != "test":
			for index, label in enumerate(self.window_sentences_labels):
				if label in l2i:
					self.window_sentences_labels[index] = l2i[label]
				else:
					self.window_sentences_labels[index] = l2i[UNKNOWN]

	def parse_sentences(self, convert_digits=True, to_lower=True):
		# parse by spaces if post, if ner parse by tab.
		delimiter = ' ' if self.is_pos else '\t'
		current_sentence_words = list()
		current_sentence_labels = list()
		for raw in self.file:
			raw_splitted = raw.split('\n')
			raw_splitted = raw_splitted[0].split(delimiter)
			word = raw_splitted[0]
			if word != '':
				# convert all chars to lower case.
				if to_lower:
					word = word.lower()
				# if we want to convert each digit to be DG for similarity, '300' = '400'.
				if convert_digits:
					word = re.sub('[0-9]', 'DG', word)
				if self.data_kind != "test":
					label = raw_splitted[1]
					current_sentence_labels.append(label)
				current_sentence_words.append(word)
			else:
				full_sentence_words = [START, START] + current_sentence_words + [END, END]
				self.sentences_words.append(full_sentence_words)

				if self.data_kind != "test":
					full_sentence_labels = [START, START] + current_sentence_labels + [END, END]
					self.sentences_labels.append(full_sentence_labels)

				current_sentence_words.clear()
				current_sentence_labels.clear()

	def get_sentences(self):
		return self.window_sentences

	def get_labels(self):
		return self.window_sentences_labels

	def get_f2i(self):
		if not self.F2I:
			self.F2I = {f: i for i, f in
						enumerate(list(sorted(set([w for sublist in self.sentences_words for w in sublist]))))}
			self.F2I[UNKNOWN] = len(self.F2I)
		return self.F2I

	def get_l2i(self):
		if not self.L2I:
			self.L2I = {l: i for i, l in
						enumerate(list(sorted(set([w for sublist in self.sentences_labels for w in sublist]))))}
			self.L2I[UNKNOWN] = len(self.L2I)
		return self.L2I

	def get_i2f(self):
		i2f = {i: l for l, i in self.F2I.items()}
		return i2f

	def get_i2l(self):
		i2l = {i: l for l, i in self.L2I.items()}
		return i2l


if __name__ == '__main__':
	p = Parser(window_size=5, data_name='ner')
	p.parse_sentences()
	p.create_windows_list_from_sentences()
	p.convert_sentences_windows_to_indexes()
