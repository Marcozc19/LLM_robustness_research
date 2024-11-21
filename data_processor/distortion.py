import pandas as pd
import random
from SoundsLike.SoundsLike import Search
import nltk
from nltk import pos_tag, word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

class DistortionProcessor:
    def __init__(self, data, distortion_types, distortion_percentage=0.3):
        self.data = data
        self.distortion_types = distortion_types
        self.distortion_percentage = distortion_percentage

        # Dictionary mapping distortion names to functions
        self.distortion_dict = {
            "character": self.character,
            "sentence": lambda data: self.sentence(data, False),
            "sentence random": lambda data: self.sentence(data, True),
            "phonetic": self.phonetic,
        }

    def apply_distortions(self):
        """Applies the specified distortions from the distortion_types list."""
        if not self.distortion_types: return self.data
        for distortion_name in self.distortion_types:
            if distortion_name not in self.distortion_dict:
                raise ValueError(f"Distortion type '{distortion_name}' not supported. Please choose from {list(self.distortion_dict.keys())}.")
            else:
                distortion_function = self.distortion_dict[distortion_name]
                self.data = distortion_function(self.data)
        print("Distortion applied")
        return self.data

    def get_data(self):
        return self.data

    def character(self, data):
        def distort_word(word):
            if len(word) > 1:
                i = random.randint(0, len(word) - 1)
                return word[:i] + random.choice('abcdefghijklmnopqrstuvwxyz') + word[i+1:]
            return word
        
        def apply_distortion(sentence):
            words = sentence.split()
            num_to_distort = max(1, int(len(words) * self.distortion_percentage))
            indices_to_distort = random.sample(range(len(words)), num_to_distort)
            for idx in indices_to_distort:
                words[idx] = distort_word(words[idx])
            return ' '.join(words)
        print("============== Distorting data - character =================")
        data['question'] = data['question'].apply(apply_distortion)
        print("============== Finished distorting data - character =================")
        return data

    def sentence(self, data, random_order: bool):
        def apply_reorder(sentence):
            words = sentence.split()
            if random_order:
                random.shuffle(words)  # Shuffle the list of words
            else:
                random_seed = random.randint(0, len(words) - 1)
                print("random seed: ", random_seed)
                words =words[random_seed:] + words[:random_seed]
            return ' '.join(words) 
        print("============== Distorting data - sentence =================")
        data['question'] = data['question'].apply(apply_reorder)
        print("============== Finished distorting data - sentence =================")
        return data

    def phonetic(self, data):
        def replace_phonetic(word):
            try:
                phonetic_options = Search.closeHomophones(word)
                if phonetic_options:
                    return random.choice(phonetic_options)
            except ValueError as e:
                # Log the specific error
                print(f"ValueError for word '{word}': {e}")
            except Exception as e:
                # Catch other unexpected exceptions
                print(f"Unexpected error for word '{word}': {e}")
            return None
        
        def apply_distortion(sentence):
            words = sentence.split()
            num_to_distort = max(1, int(len(words) * self.distortion_percentage))
            indices_to_distort = set()  # Track modified indices to avoid duplicate distortions
            
            while len(indices_to_distort) < num_to_distort and len(indices_to_distort) < len(words):
                idx = random.choice(range(len(words)))
                if idx not in indices_to_distort:
                    modified_word = replace_phonetic(words[idx])
                    if modified_word:  # Apply only if a phonetic alternative is available
                        words[idx] = modified_word
                        indices_to_distort.add(idx)
            
            return ' '.join(words)
        print("================ Distorting data - phonetic =================")
        data['question'] = data['question'].apply(apply_distortion)
        print("================ Finished distorting data - phonetic =================")
        return data
        

    def grammar(self, data):
        def remove_words(sentence, remove_pos):
            """
            Remove specific parts of speech from a sentence.

            :param sentence: The input question sentence.
            :param remove_pos: A list of POS tags to remove.
            :return: Modified sentence with specified POS tags removed.
            """
            # Tokenize the sentence
            words = word_tokenize(sentence)
            
            # Tag each word with its part of speech
            tagged_words = pos_tag(words)
            
            # Rebuild the sentence, skipping words with the specified POS tags
            filtered_words = [word for word, pos in tagged_words if pos not in remove_pos]
             
            return ' '.join(filtered_words)

        remove_pos = ['IN', 'VBZ']  # Removing prepositions and verbs (IN = preposition, VBZ = verb, 3rd person singular)
        data['question'] = data['question'].apply(lambda x: remove_words(x, remove_pos))
        return data

# Example usage
if __name__ == "__main__":
    example_data = pd.DataFrame({
        'question': ['How can I learn Python?', 'What is natural language processing?']
    })

    # Specify the types of distortions you want to apply
    distortions_to_apply = ["grammar"]
    
    processor = DistortionProcessor(example_data, distortions_to_apply, distortion_percentage=0.3)
    distorted_data = processor.apply_distortions()
    
    print(distorted_data)
