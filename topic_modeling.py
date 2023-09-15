import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import regex as re

nlp = spacy.load('en_core_web_sm', disable = ['parser', 'ner'])
my_stop_words = stopwords.words('english')
english_verbs = ['accept','achieve','add','admire','admit','adopt','advise','agree','allow','announce','appreciate','approve','argue','arrive','ask','assist','attack',
                 'bake','beg','behave','boil','borrow','brush','bury','call','challenge','change','chase','cheat','cheer','chew','clap','clean','collect','compare',
                 'complain','confess','construct','control','copy','count','create','cry','cycle','damage','dance','deliver','destroy','divide','drag','earn','employ',
                 'encourage','enjoy','establish','estimate','exercise','expand','explain','fry','gather','greet','guess','harass','hate','help','hope','identify',
                 'interrupt','introduce','irritate','joke','jump','kick','kill','kiss','laugh','lie','like','listen','love','marry','measure','move','murder','need',
                 'obey','offend','offer','open','paint','park','phone','pick','play','pray','print','pull','punch','punish','purchase','push','question','race','relax',
                 'remember','reply','retire','return','rub','scold','select','smoke','snore','stare','start','study','talk','thank','travel','trouble','type','use',
                 'visit','wait','walk','want','warn','wink','worry','yell','be','beat','become','begin','bet','bite','break','bring','build','burn','buy','catch',
                 'choose','come','cut','dig','do','dream','drink','drive','eat','fall','feel','fight','find','fly','forget','forgive','get','give','go','grow','hang',
                 'have','hear','hide','hit','hold','hurt','keep','know','learn','leave','lend','lose','make','meet','pay','put','read','ride','ring','run','say','see',
                 'sell','send','sing','sleep','speak','stand','sweep','swim','take','teach','tear','tell','think','throw','understand','wake','wear','weep','win','write']
n_features = 1500
n_topics = 8
n_top_words = 10

def lemmatize(text):
    doc = nlp(text)
    lemmatized_words = [token.lemma_ for token in doc]
    return ' '.join(lemmatized_words)

def display_topics(model, feature_names, n_top_words):
    for topic_index, topic in enumerate(model.components_):
        print("Topic %d: ", topic_index)
        print(" ".join([feature_names[i] for i in topic.argsort()[: -n_top_words -1:-1]]))

def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 4, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 20})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=30)
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()

#load dataset
df = pd.read_csv('dataset.csv')

# preprocessing
# create list of stopwords and add other words to be removed from dataset
my_stop_words.extend(['game', 'gaming', 'videogame', 'player',  'wa', 'would', 'et', 'al', 'ii', 'also', 'one', 'much', 'could','may','within', 'usually', 'moreover',
                      'current', 'never', 'even', 'must', 'really', 'juul', 'smith'])
#my_stop_words.extend(['game', 'gaming', 'videogame', 'player',  'wa', 'would', 'et', 'al', 'ii', 'also', 'one', 'character', 'world', 'time', 'aarseth', 'shepard'])
for verb in english_verbs:
    my_stop_words.append(verb)
my_stop_words
# lemmatize
df['content'] = df['content'].apply(lemmatize)
# remove digits
df['content'] = df['content'].str.replace(r'\d+', ' ', regex=True)
# remove punctuation
df['content'] = df['content'].str.replace(r'[^\w\s]+', ' ', regex=True)

documents = df['content']

# create DTM to feed to LDA
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words=my_stop_words, ngram_range=(1,3)) # matrix of token counts
dtm = tf_vectorizer.fit_transform(documents) # document-term matrix
vocab = tf_vectorizer.get_feature_names_out() # list of top 1000 words that appear in the documents

# creating LDA model
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=20, learning_method='online', learning_offset=50, random_state=0)
lda.fit(dtm)

# print and plot topics
display_topics(lda, vocab, n_top_words)
plot_top_words(lda, vocab, n_top_words, "")

# get topic distribution for each document
T=lda.fit_transform(dtm)
colnames = ["Topic "+str(i+1) for i in range(lda.n_components)]
docnames = df['date'] #[str(i) for i in range(len(documents))]
df_T = pd.DataFrame(np.round(T, 2), columns=colnames, index=docnames)

# get topic distribution for each issue (by date)
piv_table = pd.pivot_table(df_T, index=['date'], values = ['Topic 0','Topic 1','Topic 2','Topic 3','Topic 4','Topic 5','Topic 6','Topic 7'], aggfunc='mean')

# print line graph of topics 0-3
ax=piv_table.iloc[:,:4].plot(kind='line', linestyle = '-', figsize = (15, 10))
plt.title('Topic Evolution in Game Studies (2001-2023)')
plt.ylabel('topics')
plt.grid(True)
ax.set_xticks(np.arange(len(piv_table.index)))
plt.xticks(labels = piv_table.index, rotation=90)
plt.show()

# print line graph of topics 4-7
ax=piv_table.iloc[:,4:].plot(kind='line', linestyle = '-', figsize = (15, 10))
plt.title('Topic Evolution in Game Studies (2001-2023)')
plt.grid(True)
ax.set_xticks(np.arange(len(piv_table.index)))
plt.xticks(labels = piv_table.index, rotation=90)
plt.show()

# save df_T and piv_table
df_T.to_excel("tf_topic distribution per document.xlsx")
piv_table.to_excel("piv_table distribution per date.xlsx")