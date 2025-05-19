
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



## upload_file():

## download_file():

## main: