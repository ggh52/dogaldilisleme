from gensim.models import Word2Vec

# Model dosyalarını yüklemek
model_1 = Word2Vec.load("models/lemmatized_model_cbow_window2_dim100.model")
model_2 = Word2Vec.load("models/stemmed_model_skipgram_window4_dim100.model")
model_3 = Word2Vec.load("models/lemmatized_model_skipgram_window2_dim300.model")

def check_vocabulary(model, model_name):
    print(f"\n{model_name} Modeli Kelime Dağarcığı:")
    print(f"Toplam kelime sayısı: {len(model.wv.key_to_index)}")
    print("İlk 10 kelime:", list(model.wv.key_to_index.keys())[:10])
    print("'docker' kelimesi var mı?", 'docker' in model.wv.key_to_index)

# Kelime dağarcığını kontrol et
check_vocabulary(model_1, "Lemmatized CBOW Window 2 Dim 100")
check_vocabulary(model_2, "Stemmed Skipgram Window 4 Dim 100")
check_vocabulary(model_3, "Lemmatized Skipgram Window 2 Dim 300")

# 'python' kelimesi ile en benzer 3 kelimeyi ve skorlarını yazdırmak
def print_similar_words(model, model_name):
    try:
        similarity = model.wv.most_similar('docker', topn=3)
        print(f"\n{model_name} Modeli - 'docker' ile En Benzer 3 Kelime:")
        for word, score in similarity:
            print(f"Kelime: {word}, Benzerlik Skoru: {score}")
    except KeyError as e:
        print(f"\n{model_name} Modeli - Hata: {str(e)}")

# 3 model için benzer kelimeleri yazdır
print_similar_words(model_1, "Lemmatized CBOW Window 2 Dim 100")
print_similar_words(model_2, "Stemmed Skipgram Window 4 Dim 100")
print_similar_words(model_3, "Lemmatized Skipgram Window 2 Dim 300")
