# 2. Configurazione MinIO e caricamento del dataset MNIST:

## 2.1 Installazione e configurazione server MinIO

Per gestire in modo efficiente lo storage dei dati all’interno del progetto, abbiamo configurato un **server MinIO**, **un sistema di storage compatibile con lo standard Amazon S3**, che offre un’interfaccia per la memorizzazione e il recupero di oggetti.

- MinIO è stato installato su una macchina virtuale separata (ospitata sul cloud INFN) dotata di IP pubblico:
131.154.98.45
- Per l’installazione, è stata seguita la guida ufficiale disponibile al seguente repository GitHub: https://github.com/LucaPacioselli/MinIO-S3-setup/tree/main


## 2.2 Download del dataset MNIST:

Successivamente, si è provveduto a scaricare localmente il dataset MNIST, uno dei dataset di riferimento per il training e la validazione di modelli di classificazione di immagini scritte a mano. Il download è stato effettuato dal **nodo master** del cluster con i seguenti comandi:

<pre lang="markdown">

mkdir -p mnist
cd mnist

wget https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz

gunzip *.gz


</pre>

Questi comandi:
- creano una cartella **mnist/**
- scaricano i file compressi .gz contenenti immagini e etichette di training e test,e infine li decomprimono per ottenere i file **.ubyte.**


## 2.3  Caricamento del dataset MNIST su MinIO:
Mediante i seguenti comandi:

<pre lang="markdown">

python3 buckethandler.py --endpoint https://minio.131.154.98.45.myip.cloud.infn.it --bucket datasets upload --file mnist/train-images-idx3-ubyte
python3 buckethandler.py --endpoint https://minio.131.154.98.45.myip.cloud.infn.it --bucket datasets upload --file mnist/train-labels-idx1-ubyte
python3 buckethandler.py --endpoint https://minio.131.154.98.45.myip.cloud.infn.it --bucket datasets upload --file mnist/t10k-images-idx3-ubyte
python3 buckethandler.py --endpoint https://minio.131.154.98.45.myip.cloud.infn.it --bucket datasets upload --file mnist/t10k-labels-idx1-ubyte

</pre>

Il dataset è astato reso disponibile in modo centralizzato, i file sono stati caricati su MinIO all'interno di un bucket chiamato **datasets**, utilizzando lo script Python chiamato **buckethandler.py** descritto approfonditamente nel README.md presente nella cartella **utils**.



# 3. Accesso dei pod del cluster kubernetes al server MinIO:

Per permettere ai pod all’interno del cluster Kubernetes di accedere al server MinIO in modo sicuro, è stato creato un secret contenente le credenziali d’accesso:

<pre lang="markdown">

kubectl -n default create secret generic minio-credentials \
  --from-literal=AWS_ACCESS_KEY_ID=access_id \
  --from-literal=AWS_SECRET_ACCESS_KEY=access_key \
  --from-literal=AWS_ENDPOINT_URL=https://minio.131.154.98.45.myip.cloud.infn.it \
  --from-literal=AWS_REGION=us-east-1

</pre>

lo script crea un oggetto di tipo Secret nel namespace default, di nome minio-credentials;
All’interno del secret vengono memorizzate le variabili di ambiente richieste dai client;
Queste credenziali possono poi essere montate o referenziate all’interno dei pod Kubernetes per permettere accesso autenticato al bucket su MinIO

Dal nodo master del cluster Kubernetes, si è poi installato Helm (se non già presente) per poter procedere con il deploy dell’operator KubeRay:

<pre lang="markdown">

curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh

</pre>

Successivamente, si è aggiunto il repository Helm di KubeRay ed effettuata l’installazione dell’operator:

<pre lang="markdown">

helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm install kuberay kuberay/kuberay-operator

</pre>

Come già spiegato nel README.md della cartells **K8s**, l’operator KubeRay è il componente che permette di orchestrare la creazione e la gestione di cluster Ray all’interno di Kubernetes.



