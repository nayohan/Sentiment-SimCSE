from datasets import load_dataset

sub_categories =  ['emoji']#, 'emotion',  'hate', 'irony', 'offensive', 'sentiment', 'stance_abortion', 'stance_atheism', 'stance_climate', 'stance_feminist', 'stance_hillary']

path = './SentEval/data/downstream/EMOJI'
dataset_list = []
for category in sub_categories:
    dataset = load_dataset("tweet_eval", category, script_version="master")
    dataset.save_to_disk(path)
