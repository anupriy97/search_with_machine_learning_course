import argparse
import fasttext
import pandas as pd

parser = argparse.ArgumentParser(description='Process some locations and integers.')
general = parser.add_argument_group("general")
general.add_argument("--input", default="/workspace/datasets/fasttext/top_words.txt",  help="The file containing words for which synonyms is to be generated")
general.add_argument("--output", default="/workspace/datasets/fasttext/synonyms.csv", help="the file to output to")
general.add_argument("--model", default="/workspace/datasets/fasttext/title_model_normalized.bin", help="the model to be used to find synonyms")

# Setting threshold to keep only those words with cosine score above the threshold.
general.add_argument("--threshold", default=0.75, type=float, help="The minimum thresold os cosine similarity for a word to be included as synonyms (default is 0.75).")

args = parser.parse_args()
input_file = args.input
output_file = args.output
model_file = args.model
threshold = args.threshold


if __name__ == '__main__':
    # Load the synonym model trained earlier using fasttext on normalized product titles
    syn_model = fasttext.load_model(model_file)

    # Load the Top 1000 words extracted earlier using CLI command provided in the project details
    top_words_df = pd.read_csv(input_file, header=None)
    top_words_list = top_words_df[0].values.tolist()

    no_syn_list = []

    with open(output_file, 'w') as output:
        for word in top_words_list:
            nn_word_list = [word]

            nn_words = syn_model.get_nearest_neighbors(word)

            for (score, nn_word) in nn_words:
                if score < threshold:
                    break
                
                nn_word_list.append(nn_word)

            if len(nn_word_list) > 1:
                nn_word_output = ",".join(nn_word_list)
                output.write(f'{nn_word_output}\n')
            else:
                no_syn_list.append(word)

    print("Found no synonyms for %s words." % (len(no_syn_list)))
    print(no_syn_list)
