
![Logo del progetto](./img/img1.png) 
# Introduzione

Come descritto nel file README.md relativo all'overview generale, l'obiettivo del progetto è eseguire un training distribuito e il test di un modello di machine learning utilizzando il dataset MNIST.
Per gestire correttamente i dati, garantire una connessione stabile tra i vari nodi del cluster e il server di storage, nonché per poter effettuare diverse operazioni sui bucket, è stato sviluppato lo script **buckethandler.py**.
Questo script rappresenta uno strumento versatile per interagire in vari modi con un bucket esterno al cluster. In particolare, utilizza Boto3, una libreria Python che consente di comunicare con object storage compatibili con il protocollo S3, tecnologia proprietaria di AWS. Grazie a questa compatibilità, lo script può interfacciarsi anche con bucket MINIO, una soluzione open source per lo object storage che supporta lo stesso protocollo.

Di seguito vengono spiegati i metodi della classe BucketHandler utilizzati per effettuare una connessione al bucket minio(__init__), caricare il dataset MNIST (upload_file()) e scaricare il modello ottenuto dal training distribuito (downlload_file()):

## __init__:
La funzione di inizializzazione della classe BucketHandler, che viene chiamata automaticamente quando si crea un'istanza della classe, necessita che le vengano passate due stringhe, la prima realtiva all'url del backet e la seconda realtiva al nome del bucket che risponde a quel url.

Una volta salvate l'username e la password per accedere al server MINIO, viene creata una sessione di autenticazione per interagire con il bucket.
<pre lang="markdown">
try:
   self.s3_client = boto3.client(
   's3',
   endpoint_url=self.endpoint_url,
   aws_access_key_id=self.access_key,
   aws_secret_access_key=self.secret_key,
   )
except (NoCredentialsError, PartialCredentialsError) as e:
   print("Errore di autenticazione:", e)

</pre>




## upload_file():
Questa funzione Python, chiamata upload_file, serve per caricare un file su un bucket S3.
L'oggetto s3_client creato precedentemente utilizzando boto3.client('s3') utilizza il metodo upload_file, anche questo di boto3, per caricare il file specificato nel path di interesse nel bucket.

<pre lang="markdown">

def upload_file(self, file_path, object_name=None):
    try:
        object_name = object_name or file_path.split('/')[-1]
        self.s3_client.upload_file(file_path, self.bucket_name, object_name)
        print(f"File {file_path} uploaded successfully as {object_name}.")
    except Exception as e:
        print("Error uploading file:", e)

</pre>

## download_file():
Anlogamente a quanto visto per il metodo upload_file(), la funzione download_file() utilizza il metodo download_file() di boto3 per scaricare un file dal bucket.

<pre lang="markdown">

def download_file(self, object_name, file_path):
        try:
            print("Downloading file..., object_name: ", object_name)
            print("file_path: ", file_path)
            self.s3_client.download_file(self.bucket_name, object_name, file_path)
            print(f"File {object_name} downloaded successfully to {file_path}.")
        except Exception as e:
            print("Error downloading file:", e)
</pre>

## main:
Mediante l'utilizzo del modulo argparse, è possibile passare come argomenti allo script i parametri necessari per la connessione al bucket. Nel main vengono definiti i parametri che possono essere passati via linea di comando e i loro valori di default. Infine viene creata un'istanza, denominata handler, della classe S3BucketHandler.

di seguito un esempio di downloadind del dataset MNIST: 

<pre lang="markdown">
python3 buckethandler.py --endpoint  https://minio.131.154.98.45.myip.cloud.infn.it --bucket datasets upload --file mnist/train-images-idx3-ubyte
python3 buckethandler.py --endpoint  https://minio.131.154.98.45.myip.cloud.infn.it --bucket datasets upload --file mnist/train-labels-idx1-ubyte
python3 buckethandler.py --endpoint  https://minio.131.154.98.45.myip.cloud.infn.it --bucket datasets upload --file mnist/t10k-images-idx3-ubyte
python3 buckethandler.py --endpoint  https://minio.131.154.98.45.myip.cloud.infn.it --bucket datasets upload --file mnist/t10k-labels-idx1-ubyte
</pre>