import pandas as pd
from tqdm import tqdm
import zeyrek
from snowballstemmer import TurkishStemmer
import os

# Gerekli dosya yolu
INPUT_CSV = 'veri.csv'
LEMMATIZED_CSV = 'lemmatized.csv'
STEMMED_CSV = 'stemmed.csv'

# Hangi sütunlarda işlem yapılacak
TEXT_COLUMNS = ['Yetenekler', 'Egitim', 'Deneyim']

def lemmatize_text(text, analyzer):
    if pd.isna(text):
        return ''
    tokens = str(text).replace(',', ' ').split()
    lemmas = []
    for token in tokens:
        try:
            analysis = analyzer.analyze(token)
            if analysis and analysis[0] and analysis[0][0][1]:
                lemmas.append(analysis[0][0][1])
            else:
                lemmas.append(token)
        except Exception:
            lemmas.append(token)
    return ' '.join(lemmas)

def stem_text(text, stemmer):
    if pd.isna(text):
        return ''
    tokens = str(text).replace(',', ' ').split()
    stems = [stemmer.stemWord(token) for token in tokens]
    return ' '.join(stems)

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"{INPUT_CSV} bulunamadı!")
        return
    df = pd.read_csv(INPUT_CSV)
    analyzer = zeyrek.MorphAnalyzer()
    stemmer = TurkishStemmer()

    # Lemmatize
    df_lemma = df.copy()
    for col in TEXT_COLUMNS:
        tqdm.pandas(desc=f"Lemmatize {col}")
        df_lemma[col] = df_lemma[col].progress_apply(lambda x: lemmatize_text(x, analyzer))
    df_lemma.to_csv(LEMMATIZED_CSV, index=False, encoding='utf-8-sig')
    print(f"Lemmatized dosya kaydedildi: {LEMMATIZED_CSV}")

    # Stem
    df_stem = df.copy()
    for col in TEXT_COLUMNS:
        tqdm.pandas(desc=f"Stem {col}")
        df_stem[col] = df_stem[col].progress_apply(lambda x: stem_text(x, stemmer))
    df_stem.to_csv(STEMMED_CSV, index=False, encoding='utf-8-sig')
    print(f"Stemmed dosya kaydedildi: {STEMMED_CSV}")

if __name__ == "__main__":
    main() 