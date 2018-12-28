from collections import defaultdict
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics

from scipy.cluster.vq import whiten

import argparse
import logging
import codecs
import random
import numpy as np
import nltk
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')


#all_text = ' '.join(chapters)

class Trainer(object):
    """
    Reads raw dialect data and trains a classifier.
    """

    def __init__(self, model="model.pkl", data=None, verbose=False,
        classifier=None):
        """
        """
        self._model = model
        self._data = data
        self._verbose = verbose
        self._classifier = classifier
        # outcomes
        self.classes = []
        self.num_classes = 0
        self.train_X = None
        self.train_y = None
        self.vectorizer = None
        self.classifier = None
        self.pipeline = None

    def optimize(self):
        self._preprocess()
        self._build_pipeline()

        # bigger search space
        layer_size_range = range(50, 110, 10)
        solver_values = ['lbfgs', 'sgd', 'adam']
        learning_rate_init_values = [0.1, 0.01, 0.001, 0.0001]

        # define parameters whose value space needs to be searched
        param_grid = {'hidden_layer_sizes': layer_size_range,
                      'solver': solver_values,
                      'learning_rate_init': learning_rate_init_values}

        random_search = RandomizedSearchCV(estimator=self.classifier,
                                           param_distributions=param_grid,
                                           n_iter=10,
                                           cv=10,
                                           scoring='accuracy',
                                           n_jobs=2,
                                           return_train_score=True,
                                           random_state=42)

        random_search.fit(self.vectorizer.fit_transform(self.train_X), self.train_y)

        # inspect the results
        pd.options.display.max_colwidth = 100

        df = pd.DataFrame.from_dict(random_search.cv_results_)
        df.sort_values(by=["rank_test_score"])
        sys.exit(0)

    def train(self):
        """
        Preprocesses data, fits a model, and finally saves the model to a file.
        """
        self._preprocess()
        #self._build_pipeline()
        #self._fit()

    def get_chapters(self, lines):
        chapters = []
        
        for k, v in lines.items():
            for chtext in v:
                chapters.append(chtext)
            
        return chapters
        
    def get_all_text(self, chapters):
        all_text = ' '.join(chapters)
        
        return all_text
    
    def _preprocess(self):
        """
        Reads lines from the raw dialect data.
        """
        d = defaultdict(list)

        if self._data:
            data = codecs.open(self._data, "r", "UTF-8")
        else:
            logging.debug("--data not found, assuming input from STDIN")
            data = sys.stdin

        # read first line with column identifiers and ignore
        data.readline()

        for line in data:
            # skip empty lines
            line = line.strip()
            if line == "":
                continue

            X, y = line.split('","')
            d[y].append(X)
        
        feature_sets = list(self.getLexicalFeatures(d))
        feature_sets.append(self.getBagOfWords(d))
        feature_sets.append(self.getSyntacticFeatures(d))
        
        print(feature_sets)

        logging.debug("Examples per author class:")
        for k, v in d.items():
            logging.debug("%s %d" % (k, len(v)))
        logging.debug("Total messages: %d\n" %
                      sum([len(v) for v in d.values()]))

        self.classes = sorted(d.keys())
        
        self.num_classes = len(self.classes)

        l = []
        logging.debug("Samples from the data:")
        for k, values in d.items():
            logging.debug("%s\t%s" % (values[0], k))
            for value in values:
                l.append( (value, k) )

        # shuffle, just to be sure
        random.shuffle(l)
        self.train_X, self.train_y = zip(*l)

    def _build_pipeline(self):
        """
        Builds an sklearn Pipeline. The pipeline consists of a kind of
        vectorizer, followed by a kind of classifier.
        """
        # strip_accents="ascii" is a bad idea, results in lower accuracy due to the fact, that specific chars are used in specific dialects
        # analyzer="words" is a bad idea,  results in lower accuracy
        self.vectorizer = TfidfVectorizer(ngram_range=(2,2), analyzer='word')
        self.classifier = MLPClassifier(verbose=True, early_stopping=True)


        self.pipeline = Pipeline([
            ("vectorizer", self.vectorizer),
            ("clf", self.classifier)
        ])

        logging.debug(self.vectorizer)
        logging.debug(self.classifier)
        logging.debug(self.pipeline)

    def _fit(self):
        """
        Fits a model for the preprocessed data.
        """
        self.pipeline.fit(self.train_X, self.train_y)

    def save(self):
        """
        Save the whole pipeline to a pickled file.
        """
        from sklearn.externals import joblib
        joblib.dump(self.pipeline, self._model)
        logging.debug("Classifier saved to '%s'" % self._model)
        
    def getLexicalFeatures(self, lines):
        """
        Compute feature vectors for word and punctuation features
        """
        
        chapters = self.get_chapters(lines)
        
        num_chapters = len(chapters)
        fvs_lexical = np.zeros((len(chapters), 3), np.float64)
        fvs_punct = np.zeros((len(chapters), 3), np.float64)
        for e, ch_text in enumerate(chapters):
            # note: the nltk.word_tokenize includes punctuation
            tokens = nltk.word_tokenize(ch_text.lower())
            words = word_tokenizer.tokenize(ch_text.lower())
            sentences = sentence_tokenizer.tokenize(ch_text)
            vocab = set(words)
            words_per_sentence = np.array([len(word_tokenizer.tokenize(s))
                                           for s in sentences])
    
            # average number of words per sentence
            fvs_lexical[e, 0] = words_per_sentence.mean()
            # sentence length variation
            fvs_lexical[e, 1] = words_per_sentence.std()
            # Lexical diversity
            fvs_lexical[e, 2] = len(vocab) / float(len(words))
    
            # Commas per sentence
            fvs_punct[e, 0] = tokens.count(',') / float(len(sentences))
            # Semicolons per sentence
            fvs_punct[e, 1] = tokens.count(';') / float(len(sentences))
            # Colons per sentence
            fvs_punct[e, 2] = tokens.count(':') / float(len(sentences))
    
        # apply whitening to decorrelate the features
        fvs_lexical = whiten(fvs_lexical)
        fvs_punct = whiten(fvs_punct)
    
        return fvs_lexical, fvs_punct
    
    def getBagOfWords(self, lines):
        """
        Compute the bag of words feature vectors, based on the most common words
         in the whole book
        """
        #for k, v in lines.items():
        #    for e, ch_text in enumerate(v):
                
        chapters = self.get_chapters(lines)
        
        all_text = self.get_all_text(chapters)
                
        # get most common words in the whole book
        NUM_TOP_WORDS = 10
        all_tokens = nltk.word_tokenize(all_text)
        fdist = nltk.FreqDist(all_tokens)
        vocab = list(fdist.keys())[:NUM_TOP_WORDS]

        # use sklearn to create the bag for words feature vector for each chapter
        vectorizer = CountVectorizer(vocabulary=vocab, tokenizer=nltk.word_tokenize)
        fvs_bow = vectorizer.fit_transform(chapters).toarray().astype(np.float64)
    
        # normalise by dividing each row by its Euclidean norm
        fvs_bow /= np.c_[np.apply_along_axis(np.linalg.norm, 1, fvs_bow)]
    
        return fvs_bow

    def getSyntacticFeatures(self, lines):
        """
        Extract feature vector for part of speech frequencies
        """
        
        chapters = self.get_chapters(lines)
        
        all_text = self.get_all_text(chapters)
        
        def token_to_pos(ch):
            tokens = nltk.word_tokenize(ch)
            return [p[1] for p in nltk.pos_tag(tokens)]
    
        chapters_pos = [token_to_pos(ch) for ch in chapters]
        pos_list = ['NN', 'NNP', 'DT', 'IN', 'JJ', 'NNS']
        fvs_syntax = np.array([[ch.count(pos) for pos in pos_list]
                               for ch in chapters_pos]).astype(np.float64)
    
        # normalise by dividing each row by number of tokens in the chapter
        fvs_syntax /= np.c_[np.array([len(ch) for ch in chapters_pos])]
    
        return fvs_syntax
class Predictor(object):
    """
    Predicts the dialect of text, given a trained model.
    """

    def __init__(self, model="model.pkl"):
        """
        """
        self._model = model
        self._load()

    def _load(self):
        """
        Loads a model that was previously trained and saved.
        """
        from sklearn.externals import joblib
        self.pipeline = joblib.load(self._model)
        logging.debug("Loading model pipeline from '%s'" % self._model)

    def predict(self, samples, label_only=False):
        """
        Predicts the class (=dialect) of new text samples.
        """
        predictions = []

        for sample in samples:
            if label_only:
                predictions.append(self.pipeline.predict([sample])[0])
            else:
                sample = sample.strip().split(",")[1]  # column 0 is the index
                predictions.append((sample, self.pipeline.predict([sample])[0]))

        return predictions

    def evaluate(self, samples):
        """
        Evaluates the classifier with gold labelled data.
        """
        test_y = []
        test_X = []
        for sample in samples:
            sample = sample.strip()
            X,y = sample.split('","')
            test_y.append(y)
            test_X.append(X)

        logging.debug("Number of gold samples found: %d" % len(test_y))

        predictions = self.predict(test_X, label_only=True)
        report = metrics.classification_report(test_y, predictions, target_names=None)
        logging.info(report)
        
def parse_cmd():
    parser = argparse.ArgumentParser(
        description="train a classifier for dialect data and use it for predictions")

    parser.add_argument(
        "-m", "--model",
        type=str,
        required=False,
        help="if --train, then save model to this path. If --predict, use saved model at this path."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        required=False,
        help="write verbose output to STDERR (default: False)"
    )

    mode_options = parser.add_mutually_exclusive_group(required=True)
    mode_options.add_argument(
        "--train",
        action="store_true",
        required=False,
        help="train a new model and save to the path -m/--model"
    )
    mode_options.add_argument(
        "--predict",
        action="store_true",
        required=False,
        help="predict classes of new samples, write predicted classes to STDOUT"
    )
    mode_options.add_argument(
        "--evaluate",
        action="store_true",
        required=False,
        help="evaluate trained model, write report to STDOUT. If --evaluate, data in --samples is assumed to include the gold label"
    )
    mode_options.add_argument(
        "--split",
        action="store_true",
        required=False,
        help="split data from --data to train and test sets and then exit"
    )
    mode_options.add_argument(
        "--optimize",
        action="store_true",
        required=False,
        help="Run randomized search to optimize the parameters"
    )

    train_options = parser.add_argument_group("training parameters")

    train_options.add_argument(
        "--data",
        type=str,
        required=False,
        help="path to file with raw dialect data, UTF-8. If --data is not given, input from STDIN is assumed"
    )


    predict_options = parser.add_argument_group("prediction parameters")

    predict_options.add_argument(
        "--samples",
        type=str,
        required=False,
        help="Path to file containing samples for which a class should be predicted. If --samples is not given, input from STDIN is assumed"
    )

    split_options = parser.add_argument_group("split parameters")

    split_options.add_argument(
        "--shuffle",
        type=bool,
        default=True,
        required=False,
        help="flag to force data shuffling when splitting"
    )

    split_options.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        required=False,
        help="a number between 0 and 1 to specify the percentage of a test subset, e.g. 0.1 would mean 10 percent i.e. 90/10 split"
    )

    args = parser.parse_args()

    return args

def main():
    args = parse_cmd()
    
    # set up logging
    if args.verbose:
        level = logging.DEBUG
    elif args.evaluate:
        level = logging.INFO
    else:
        level = logging.WARNING
        
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    
    if args.split:
        logging.debug("splitting file %s into 2 files" % args.data)

        lines = []
        if args.data:
            with codecs.open(args.data, 'r', encoding="utf-8") as f:
                lines = f.readlines()

            header_line = lines[0]
            res = train_test_split(lines[1:], shuffle=args.shuffle, test_size=args.test_size)

            with codecs.open("{}.split{}".format(args.data, (int)((1 - args.test_size)*100)), 'w', encoding="utf-8") as out:
                out.write(header_line)
                out.writelines(res[0])

            with codecs.open("{}.split{}".format(args.data, (int)(args.test_size*100)), 'w', encoding="utf-8") as out:
                out.write(header_line)
                out.writelines(res[1])

        #splitting is a separate step, so we stop after it is done
        return
    
    if args.optimize:
        t = Trainer(model=args.model,
                    data=args.data,
                    verbose=args.verbose
                    )
        t.optimize()
        return

    if args.train:
        t = Trainer(model=args.model,
                    data=args.data,
                    verbose=args.verbose
                    )
        t.train()
        t.save()
        
    else:
        p = Predictor(model=args.model)
        if args.samples:
            input_ = codecs.open(args.samples, "r", "UTF-8")
        else:
            logging.debug("--samples not found, assuming input from STDIN")
            input_ = sys.stdin

        # read first line and ignore, column names
        input_.readline()

        if args.evaluate:
            p.evaluate(samples=input_)
        else:
            predictions = p.predict(samples=input_, label_only=True)
            print ("Id,Prediction")
            for index, prediction in enumerate(predictions):
                print ("%s,%s" % (index+1, prediction))
    
if __name__ == '__main__':
    main()