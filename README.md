# Filtro anti-hater per social network

## Descrizione e obiettivi del progetto
Questo progetto, realizzato durante il Master in Data Science di Profession AI, ha come obiettivo l'automatizzazione del processo di moderazione dei commenti tossici su un forum online, affrontando l’inefficacia dei metodi tradizionali. A tale scopo viene costruito un modello di Deep Learning con layer ricorrenti per classificare i commenti in sei categorie di tossicità, permettendo un filtraggio in tempo reale (**classificazione multilabel**). Il modello viene sviluppato in **Python** su ambiente **Google Colab**, sfruttando la libreria **Keras** all'interno di **TensorFlow**.<br>
Il fine è quindi ridurre il carico della moderazione manuale, migliorare l’accuratezza nel rilevamento dei contenuti offensivi e garantire un ambiente online più sicuro e inclusivo. 

## Dataset
Il dataset `Filter_Toxic_Comments_dataset`, scaricabile direttamente all'interno del notebook, è costituito da 159571 samples e contiene la colonna `comment_text` con i messaggi degli utenti e le seguenti colonne con le labels (variabili dummy 0/1) che rappresentano la dannosità del commento:
- **`toxic`**
- **`severe_toxic`**
- **`obscene`**
- **`threat`**
- **`insult`**
- **`identity_hate`**
  
Inoltre è presente anche la colonna `sum_injurious` che somma tutte le eventuali dannosità presenti per dare un'idea di quanto sia complesivamente dannoso il commento (da 0 a 6).

## Struttura del progetto

1. **Analisi descrittiva**:
   - Il 90 % dei commenti risulta non dannoso mentre il restante 10 % ha almeno una label pari ad 1: dataset fortemente sbilanciato.
   - Anche le diverse labels risultano sbilanciate tra di loro:
![frequenze_labels](https://github.com/VirginioC/filtro-anti-hater-social-network/blob/main/frequenze_labels.png)
   - La matrice di co-occorrenza indica che le labels oltre ad essere sbilanciate sono spesso presenti simultaneamente: si preferisce quindi optare per l'uso di pesi personalizzati piuttosto che tecniche di undersampling/oversampling.

2. **Preprocessing dei dati**:
   - Viene preprocessato il testo eliminando l'insieme di token che non danno contributo significativo a livello semantico: conversione in lowercase, rimozione della punteggiatura, rimozione delle stopwords della lingua inglese, lemmatizzazione e rimozione degli spazi bianchi extra.
   - Creazione dell'array contenente i messaggi testuali e dell'array contenente le labels.
   - Splitting del dataset in train set (70 %), validation set (20 %) e test set (10 %).
  
3. **Machine Learning**:
   - Prima di valutare le prestazioni di modelli costituiti da layer ricorrenti si parte da un modello "baseline" di regressione logistica:
     - Uso della funzione **tf-idf** per vettorizzare il testo.
     - Addestramento e valutazione del modello di **Multilabel Logistic Regression**, con e senza parametro `class_weight="balanced"`, ricercando sul validation set i parametri che ottimizzano l'**F1-score macro**: per questo problema l'**F1-score** è la metrica più importante da valutare essendoci un consistente sbilanciamento delle classi. Si ricava poi il classification report con tutte le metriche: si ottiene in particolare, sul test set, un **F1-score macro** del **51 %** e un **F1-score weighted** del **66 %**. 
       
4. **Deep Learning**:
   - Si passa poi a valutare l'efficacia di modelli di deep learning costituiti da layer ricorrenti che sono i più adatti in presenza di corpus testuali:
     - Trasformazione del corpus testuale in sequenze tramite tokenizer di Keras: vocabolario con le 10000 parole più frequenti, lunghezza massima delle sequenze pari a 77 (90° percentile delle lunghezze nel train set) e padding alla fine delle sequenze, ottenendo per train set, validation set e test set sequenze con 77 elementi ciascuna.
     - Si sceglie di valutare tre modelli di **reti neurali ricorrenti**:
       - **`LSTM Model`**:
       - **`Bidirectional LSTM Model`**
       - **`Convolutional Bidirectional LSTM Model`**
     - Per il **training** e l'**evaluation** dei modelli vengono costruite delle specifiche funzioni dettagliatamente descritte nel notebook. In particolare, si cerca di contrastare lo sbilanciamento delle classi durante l'addestramento tramite la funzione `weighted_binary_crossentropy` che restituisce una funzione di perdita che calcola la cross-entropia binaria ponderata tra le etichette reali e le probabilità previste applicando dei pesi personalizzati per ciascuna label (calcolati come l'inverso delle frequenze relative delle classi positive delle labels normalizzate).
     - Il modello "migliore" risulta essere quello che utilizza il `Bidirectional LSTM` come layer ricorrente. Esso ha la seguente architettura:
       
       ```python
            ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
            ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
            ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
            │ embedding (Embedding)                │ (None, 77, 128)             │       1,280,000 │
            ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
            │ bidirectional (Bidirectional)        │ (None, 256)                 │         263,168 │
            ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
            │ dropout (Dropout)                    │ (None, 256)                 │               0 │
            ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
            │ dense (Dense)                        │ (None, 6)                   │           1,542 │
            └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
             Total params: 1,544,710 (5.89 MB)
             Trainable params: 1,544,710 (5.89 MB)
             Non-trainable params: 0 (0.00 B)
       ```
          
       e viene addestrato (`batch_size=32` e `epochs=20`) massimizzando l'**F1-score macro** sul validation set. Si ottiene dopo **6 epoche** (uso dell'**early stopping**):
       
       - Train F1-score macro: **81 %** - Train F1-score weighted: **82 %**
       - Validation F1-score macro: **62.13 %** - Validation F1-score weighted: **73.38 %**
       - Test F1-score macro: **58.71 %** - Test F1-score weighted: **72.77 %**

         ```python         
            Bidirectional LSTM - Classification Report - Test Set
                           precision    recall  f1-score   support
            
                    toxic       0.82      0.75      0.78      1550
             severe_toxic       0.44      0.42      0.43       158
                  obscene       0.81      0.77      0.79       826
                   threat       0.36      0.35      0.36        48
                   insult       0.68      0.68      0.68       778
            identity_hate       0.50      0.47      0.49       121
            
                micro avg       0.75      0.71      0.73      3481
                macro avg       0.60      0.57      0.59      3481
             weighted avg       0.75      0.71      0.73      3481
              samples avg       0.97      0.96      0.94      3481
            ```
         I miglioramente che si ottengono rispetto al modello baseline sono significativi, tuttavia le prestazioni nel complesso non sono del tutto soddisfacenti essendo arrivati ad ad avere dei risultati "medi" sulle metriche pesate ma "medio-bassi" sulle metriche macro, continuando ad avere difficoltà nel prevedere le labels meno frequenti. Resta, infatti, un significativo problema di **overfitting** (senza l'early stopping si sarebbe arrivati ad avere delle prestazioni davvero ottime sul training set) che non è stato possibile ridurre in maniera convincente nonostante i vari modelli e tecniche.

       - Selezionato il miglior modello, si effettua il salvataggio dei suoi pesi e del tokenizer in due file, presenti anche nel repository:
          - `tokenizer.joblib`
          - `BiLSTM.keras`
       - Questi due file possono essere direttamente scaricati e ricaricati dall'utente nell'ambiente Google Colab senza essere costretti a rieseguire l'addestramento.
      
5. **Test del prototipo di filtro anti-hater su esempi casuali**
   - Viene infine creata la funzione per il filtraggio dei commenti tossici `toxic_comments_filter` che racchiude in un'unica funzione tutte le operazioni di pulizia testuale, tokenizzazione (uso di `tokenizer.joblib`), trasformazione in sequenze "paddate" e predizione del modello (uso di `BiLSTM.keras`).
   - Si testa la funzione su alcuni esempi casuali appartenenti al test set:
     
     ```python
     Labels names:
     Index(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
             'identity_hate'],
            dtype='object')
      
     Sentence:
     buck winston fuck tell fuck tell fuck tell fuck tell fuck tell fuck tell fuck tell fuck tell fuck tell fuck tell fuck tell fuck tell fuck tell fuck tell fuck tell motherfucker ugh
      
     Predicted labels: [[1 1 1 0 1 0]]
     True labels: [1 1 1 0 1 0]
     ---------------------------------
     Sentence:
     wow thing like childish real piece work true coward
      
     Predicted labels: [[1 0 0 0 0 0]]
     True labels: [1 0 0 0 0 0]
     ---------------------------------
     ```

## Tecnologie utilizzate
- **Linguaggio**: Python
- **Ambiente di sviluppo**: Google Colab (Jupyter Notebook)
- **Librerie**:
  - `google.colab`
  - `requests`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `nltk`
  - `scikit-learn`
  - `tensorflow`
  - `keras`
  - `joblib`
  - `gdown`

## Utilizzo  
1. Scarica o clona il repository.
2. Apri il file `filtro_anti-hater_social_network.ipynb` su Google Colab o altri ambienti compatibili con Jupyter Notebook.
3. Esegui il codice passo-passo per ottenere i risultati.
4. Se vuoi evitare di riaddestrare i modelli puoi anche direttamente scaricare e ricaricare nell'ambiente Google Colab il tokenizer `tokenizer.joblib` e i pesi del modello migliore `BiLSTM.keras` e testare direttamente il prototipo con la funzione `toxic_comments_filter`.

## Autore
[Virginio Cocciaglia](https://github.com/VirginioC)

---
