
import os
import random
import converter
from collections import Counter


class Dataset:
    def __init__(
        self,
        sequence_length,
        folders,
    ):
        self.sequence_length = sequence_length

        self.folders = folders

        all_words, self.X, self.Y = self.load_svgs()

        self.uniq_words = self.get_uniq_words(all_words)

        self.word2index = {word: index for index, word in enumerate(self.uniq_words)}

        self.X = [[self.word2index[wi] for  wi in w] for w in self.X]

    def get_pairs(self):

        pairs = list(zip(self.X, self.Y))

        return pairs

    def get_svg_paths(self):
        svg_paths = []

        for i, folder in enumerate(self.folders):
            files = [file for file in os.listdir(folder)]

            for file in files:
                svg_paths.append([file.split('.')[0], os.path.join(folder, file)])

        return svg_paths

    def load_svgs(self):
        c = converter.Converter(self.sequence_length)
        svg_paths = self.get_svg_paths()
        random.shuffle(svg_paths)

        X = []
        Y = []
        all_words = []

        for svg_name, svg_path in svg_paths:
            try:
                vector = c.to_vector(c.open(svg_path))
                svg_name = svg_name.split('_')[1:][0]
                svg_name = [word for word in svg_name.split('-')]
                # svg_name += ['NULL']*(self.args.sequence_length-len(svg_name))

                X.append(svg_name)
                Y.append(vector)
                all_words.extend(svg_name)
            except:
                pass

        return all_words, X, Y

    def get_uniq_words(self, words):
        word_counts = Counter(words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return (
            torch.tensor(self.X[index]),
            torch.tensor(self.Y[index]),
        )

