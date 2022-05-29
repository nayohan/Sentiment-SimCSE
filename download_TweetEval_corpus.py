from datasets import load_dataset


sub_categories =  ['emoji', 'emotion',  'hate', 'irony', 'offensive', 'sentiment', 'stance_abortion', 'stance_atheism', 'stance_climate', 'stance_feminist', 'stance_hillary']

dataset_list = []
for category in sub_categories:
    dataset = load_dataset("tweet_eval", category, script_version="master")
    dataset_list.append(dataset)

# print(dataset_list[0]['train']['text'][11])

emoji_corpus = []
for i in range(len(dataset_list)):
    print(len(dataset_list[i]['train']['text']))
    emoji_corpus.extend(dataset_list[i]['train']['text'])

print(len(emoji_corpus))

with open("emoji_corpus.txt", "w", encoding='utf8') as f:
    for text in emoji_corpus:
        f.write("%s\n" % text)
f.close()