# Introduzione:
Una volta conclusa l'implementazione del cluster Kubernetes, e stabilita una corretta comunicazione tra i vari nodi (masternode & workernodes) ed il server MINIO, si procede con la costruzione del modello di rete neurale, utilizzato per la classificazione delle immagini del dataset MNIST.
Il progetto prevede due fasi principali: l’addestramento e la verifica del modello, gestite rispettivamente dagli script **train_mnist_ray.py** e **run_test.py**, presenti nella directory.
Il primo script (```train_mnist_ray.```) si occupa di stabilire una connessione con il server MINIO per scaricare le immagini e le etichette necessarie all’addestramento. Successivamente, i dati vengono organizzati mediante un dataloader e utilizzati per istruire il modello di rete neurale.

Il secondo script (```run_test.py```), esattamente come il primo, entra in contatto con il server MINIO, da cui però scarica le immagini e le etichette necessarie per il test e il modello generato dall'addestramento. Infine verifica il corretto funzionamento della rete neurale assegnando un valore di Accuracy alle predizioni fatte. 




## Training distribuito:
Il codice esegue un addestramento distribuito di un modello neurale sulla base MNIST, utilizzando dati salvati su un bucket MINIO S3, sfruttando Ray Train per la distribuzione del carico tra più worker. Di seguito si descrive il funzionamento delle principali funzioni presenti nello script e il loro utilizzo nel main: 

### 1. setup_minio_connection():

<pre lang="markdown">
S3_PREFIX = "s3://datasets/mninst" 
ENDPOINT  = os.getenv("AWS_ENDPOINT_URL") 
REGION    = os.getenv("AWS_REGION", "us-east-1")

s3_config = S3ClientConfig(force_path_style=True)
</pre>

Il codice configura l'accesso al bucket MINIO, definendo le variabili necessarie per la connessione.
S3ClientConfig() è fornita dalla libreria s3torchconnecto, che consente di caricare dati direttamente da un bucket compatibile con s3 come **tocrh.utils.data.Dataset**.
Nello specifico S3ClientConfig() configura un client S3 utilizzando la libreria boto3 per accedere al bucket MINIO.



### 2. _download_and_concat()
La funzione scarica i file in memoria e fornisce i contenuti combinati in un singolo oggetto bytes.

Inizialmente viene creato un dataset s3 con il metodo S3MapDataset.from_prefix(). Questo permentte di scaricare dati direttamente da un bucket compatibile con s3 ed usarli come un dataset pyTorch. 

<pre lang="markdown">
    ds = S3MapDataset.from_prefix(
        prefix,
        region=REGION,
        endpoint=ENDPOINT,
        transform=None,
        s3client_config=s3_config
    )
</pre>

Viene formata una lista di coppie del tipo ```["my-bucket", "datasets/mnist/train/part-00001" ]```che viene filtrata per eliminare tutti i file che presentano l'estensione xl.meta (metadati), e successivamente viene ordinata alfabeticamente. 

<pre lang="markdown">
    pairs = ds._dataset_bucket_key_pairs
    indices = sorted(
      [i for i, entry in enumerate(pairs) if not entry[1].endswith("xl.meta")],
      key=lambda i: pairs[i][1]
    )
</pre>

Per ogni elemento della lista "indices" viene aperto l'oggetto di quella lista, viene letto il contenuto e viene concatenato alla lista "buf". Di fatto quello che si sta facendo è una concatenazione di files binari in ram.
NB: Il motivo per cui viene fatta questa cosa, è che in molti casi datasets di grandi dimensioni vengono splittati in più parti nel bucket MINIO. Questa funzione serve per ricostruite il file binario originale così da lavorarci direttamente. (nel nostro caso non serve).





### 3. _parse_idx_images():
La funzione in questione, serve per decodificare i files binari con formato IDX in array Numpy utilizzabili. (Si noti che nella cartella "datasets/mnist" sono presenti due file: "train-images-idx3-ubyte" e "train-labels-idx1-ubyte"). IDX è un formato di file binario utilizzato per rappresentare array multidimensionali.

Il file binario MNIST ha questa struttura:
    - 4 byte: magic number (0x00000803) identifica immagine
    - 4 byte: numero di immagini
    - 4 byte: numero di righe
    - 4 byte: numero di colonne
    - tutti gli altri byte: valori che compongono le immagini

la riga:
<pre lang="markdown">
magic, n, rows, cols = struct.unpack_from(">IIII", bytestream, 0)
</pre>
decodifica i primi 16 byte e li trasforma in una tupla (identificatore_img, numero immagini, numero rows, numero cols). 

infine la riga di codice:
<pre lang="markdown">
data = np.frombuffer(bytestream, dtype=np.uint8, offset=16)
return data.reshape(n, rows, cols)
</pre>
salta i primi 16 bytes dell'header legge i dati rimanenti come array flat (0-255), di dimensioni (n * rows * cols).
Viene effettuato un reshape dell'array flat convertendolo in un tensore (n, rows, cols).


In breve viene effettuato un parsing di un file binario e viene restituito un array Numpy di dimensione (n, rows, cols).


### 4. _parse_idx_labels():
La funzione in question fa esattamente la stessa cosa di _parse_idx_images() però tenendo conto che quello che deve essere parsato è un file di etichette e non un file di immagini.

### 5. make_dataset():
La funzione sfrutta _download_and_concat, _parse_idx_images e _parse_idx_labels per creare un dataset di immagini e etichette. Esattamente quello che ci serve per implementare il training del modello.

Inizialmente viene costruito dinamicamente il path per raggiungere le imagini e le etichette nel  bucket MINIO. A seconda che si stia trattando il training set o il test set

<pre lang="markdown">
img_prefix = f"{S3_PREFIX}/{split}/{'train-images-idx3-ubyte' if split=='train' else 't10k-images-idx3-ubyte'}"
(...)
lbl_prefix = f"{S3_PREFIX}/{split}/{'train-labels-idx1-ubyte'  if split=='train' else 't10k-labels-idx1-ubyte'}"
</pre>

Ottenuti i path a cui riferirsi per il download delle immagini e delle etichette vengono chiamate le funzioni _download_and_concat, _parse_idx_images e _parse_idx_labels.

<pre lang="markdown">
imgs_bytes = _download_and_concat(img_prefix)
labs_bytes = _download_and_concat(lbl_prefix)
(...)
imgs_np = _parse_idx_images(imgs_bytes)       # shape (N,28,28)
labs_np = _parse_idx_labels(labs_bytes) 
</pre>

Infine, dai vari tensori numpy vengono creati i tensori pyTorch, pronti per essere utilizzati nel training del modello. 
Si aggiunge una dimensione al tensore di immagini, perchè pyTorch si aspetta un tensore di dimensione (batch_size, channels, height, width), e dato che le immagini sono in bianco e nero, channel dovrà essere pari ad 1.

### 6. train_loop_per_worker():

### 7. main();



## Test del modello:

