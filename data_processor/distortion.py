import pandas as pd
import random
from SoundsLike.SoundsLike import Search

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
            "grammar": self.grammar,
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

        data['question'] = data['question'].apply(apply_distortion)
        print("")
        return data

    def sentence(self, data, random_order: bool):
        def reorder_characters(word):
            if len(word) > 1:
                start_index = random.randint(0, len(word) - 1)
                if random_order:
                    return word[start_index:] + word[:start_index]
                else:
                    return word[start_index:][::-1] + word[:start_index][::-1]
            return word
        
        def apply_reorder(sentence):
            words = sentence.split()
            reordered_words = [reorder_characters(word) for word in words]
            return ' '.join(reordered_words)

        data['question'] = data['question'].apply(apply_reorder)
        return data

    def phonetic(self, data):
        def replace_phonetic(word):
            phonetic_options = Search.closeHomophones(word)
            if phonetic_options:
                return random.choice(phonetic_options)
            return word
        
        def apply_distortion(sentence):
            words = sentence.split()
            num_to_distort = max(1, int(len(words) * self.distortion_percentage))
            indices_to_distort = random.sample(range(len(words)), num_to_distort)
            for idx in indices_to_distort:
                words[idx] = replace_phonetic(words[idx])
            return ' '.join(words)

        data['question'] = data['question'].apply(apply_distortion)
        return data

    def grammar(self, data):
        def delete_random_word(sentence):
            words = sentence.split()
            if len(words) > 1:
                i = random.randint(0, len(words) - 1)
                del words[i]
            return ' '.join(words)

        data['question'] = data['question'].apply(delete_random_word)
        return data

# Example usage
if __name__ == "__main__":
    example_data = pd.DataFrame({
        'question': ['How can I learn Python?', 'What is natural language processing?']
    })

    # Specify the types of distortions you want to apply
    distortions_to_apply = ["character", "phonetic", "grammar"]
    
    processor = DistortionProcessor(example_data, distortions_to_apply, distortion_percentage=0.3)
    distorted_data = processor.apply_distortions()
    
    print(distorted_data)
