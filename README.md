# Introduzione:
Il progetto "Distributed-Training" ha l’obiettivo di seguire tutti i passaggi necessari per addestrare un modello di rete neurale, distribuendo il processo di training su un cluster Kubernetes composto da più nodi.
Nel progetto sono implementati tutti gli step fondamentali per distribuire un processo di machine learning su un cluster, partendo dalla configurazione infrastrutturale del cluster Kubernetes e del cluster Ray, passando per l’implementazione delle comunicazioni tra il cluster e un server MinIO esterno, fino ad arrivare alla fase di addestramento e validazione del modello di machine learning.

L’infrastruttura del progetto è composta da:
 - Un cluster Kubernetes formato da 3 nodi: un nodo master e due nodi worker.
 - Un cluster Ray con 8 worker distribuiti sui due nodi worker del cluster Kubernetes.
 - Un server MinIO, responsabile della gestione del dataset MNIST e del modello addestrato.

 ![Logo del progetto](./img/img.png) 


 # Implementazione Infrastrutturale: 
 ## 1. Cluster Kubernetes:
 In prima istanza sono state create tre macchine virtuali su di un Server privato, con i seguenti hostname:
 - **masternode** (192.168.122.33) con: 10 CPUs & 22Gb di Ram
 - **workernode1** (192.168.122.22) con: 8 CPUs & 17Gb di RAM
 - **workernode2** (192.168.122.94) con: 8 CPUs & 17Gb di RAM

Il masternode svolgerà il ruolo di control plane del cluster Kubernetes, mentre i due nodi worker1 e worker2 ricopriranno il ruolo di workersnode.

### 1.1 Installazione di Kubernetes con rke2 & avvio del servizio:
- Per installare Kubernetes sul cluster di macchine virtuali è stato utilizzato il tool rke2, dando il comando:  (spiegazione breve di rke2). 
```curl -sfL https://get.rke2.io/ | INSTALL_RKE2_TYPE=server sh -```

- Per abilitare il servizio utilizzando rke2, e successivamente metterlo in atto vengono dati i due comandi: 

    - ```systemctl enable rke2-server.service```
    - ```systemctl start rke2-server.service```

Una volta installato il servizio, il kubeconfig (spiegare cosa sia e a cosa serve) viene salvato nella directory seguente: ```/etc/rancher/rke2/rke2.yaml```
E' necessario a questo punto esportare la variabile d'ambiente KUBECONFIG (perchè?) con il cpmando: 

- ```export KUBECONFIG=/etc/rancher/rke2/rke2.yaml```

Avendo esportato la variabile d'ambiente KUBECONFIG, è ora possibie eseguoire tutti i comandi di kubectl per interagire con il cluster senza dover specificare ogni volta il path del kubeconfig, passando da: 

- ```/var/lib/rancher/rke2/bin/kubectl get nodes```

a 

- ```kubectl get nodes```

In queto caso l'output ottenuto è il seguente: forsae qui è meglio metterci una piccola foto:
 ``` NAME STATUS ROLES AGE VERSION masternode0.localdomain Ready control-plane,etcd,master 17m v1.31.8+rke2r1 ```

Come è possibile notare dall'output il nodo master è stato creato correttamente, ed è pronto all'uso come possiamo evincere dal suo status "ready".

E' possibile andare a vedere tutti i PODs del cluster con il comando: 

- ```kubectl get pods -A```
Ottenendo così il seguente output: 
 ![output1](./img/img.png) 



## Installazione di rke2 sui nodi worker:
A questo punto è necessario installare rke2 sui nodi worker, (perchè? mi da delle migliorie?) eseguendo il comando: 

```curl -sfL https://get.rke2.io/ | INSTALL_RKE2_TYPE=agent sh -```

Successivamente viene creata la cartella rke2: ```sudo mkdir -p /etc/rancher/rke2```
e il file di configurazione config.yaml: 

``` sudo tee /etc/rancher/rke2/config.yaml > /dev/null <<EOF ````

server: https://192.168.122.33:9345
token: K10aca9023f14ec740f69a6f15659aac21d56b8631d93f2b417c51111fd89e640cf::server:1a7560a20238099cd225b0aa3def7cb6
EOF ```


Creato il file, e' necessario abilitare rke2 agent e farlo partire: 

- ``` systemctl enable rke2-agent.service ```
- ``` systemctl start rke2-agent.service ```

Tornando sul control plane possiamo vedere i nodi del cluster: 
 ![output1](./img/)  immagine dei vari nodi:

 Che sono entrambi Ready (NON banale, perche' rke2 si occupa anche della configurazione della rete)
In questo caso, rke2 utilizza Canal come plugin di rete. 


### 1.2 Installazione di Helm:
Sul nodo control plane del cluster k8s installiamo helm (da dare una letta su cosa e')

 <pre lang="markdown">curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh</pre>

Con helm installiamo il kuberay operator (da studiare): 
<pre lang="markdown">
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm install kuberay-operator kuberay/kuberay-operator --version 1.3.0
</pre>

Completato il comando ed eseguendo di nuovo il comando per recupare i pods, vedermeo il pod che gestisce 
il kuberay-operator: 
immagine di output.
![output1](./img/)


# 2. Configurazione MinIO e caricamento del dataset MNIST:
Abbiamo configuto MINIO su un'altra macchina virtuale (sempresu INFN cloud con IP pubblico 131.154.98.45)

seguendo questo guida nel repo:
https://github.com/LucaPacioselli/MinIO-S3-setup/tree/main

E fatto partire il minio server utilizzando Docker compose.

Si è scaricato il dataset MNIST localmente con i comandi: 

<pre lang="markdown">
mkdir -p mnist
cd mnist

wget https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz

gunzip *.gz
</pre>

Ed si è caricato il dataset su MinIO utilizzando i seguenti comandi: 
<pre lang="markdown">
python3 buckethandler.py --endpoint  https://minio.131.154.98.45.myip.cloud.infn.it --bucket datasets upload --file mnist/train-images-idx3-ubyte
python3 buckethandler.py --endpoint  https://minio.131.154.98.45.myip.cloud.infn.it --bucket datasets upload --file mnist/train-labels-idx1-ubyte
python3 buckethandler.py --endpoint  https://minio.131.154.98.45.myip.cloud.infn.it --bucket datasets upload --file mnist/t10k-images-idx3-ubyte
python3 buckethandler.py --endpoint  https://minio.131.154.98.45.myip.cloud.infn.it --bucket datasets upload --file mnist/t10k-labels-idx1-ubyte
</pre>

Questo non so cosa faccia: 
<pre lang="markdown">
kubectl -n default create secret generic minio-credentials --from-literal=AWS_ACCESS_KEY_ID=access_id --from-literal=AWS_SECRET_ACCESS_KEY=access_key --from-literal=AWS_ENDPOINT_URL=https://minio.131.154.98.45.myip.cloud.infn.it --from-literal=AWS_REGION=us-east-1
</pre>


sul nodo master: 

<pre lang="markdown">
curl -fsSL -o get_helm.sh https://raw.githubusercontent.comhelm/helm/main/scripts/get-helm-3 chmod 700 get_helm.sh
./get_helm.sh
</pre>

<pre lang="markdown">
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm install kuberay kuberay/kuberay-operator
</pre>


Scaletta dei vari steps fatti che devono esere ben spiegati per esame:
- [ ] Spiega BuildingLuster.txt e DockerFile
- [ ] Spiega i files che si torvano dentro il K8s folder
    - [ ] Raycluster.yaml
    - [ ] Rayjob.yaml
    - [ ] Service-account.yaml
- [ ] Spiega il file dentro utils, buckethandler.py
- [ ] Spiega il Dataset e la connessione al db minio, magari il buckethandler.py
- [ ] Spiega il file train_mnist.py
- [ ] Spiega il run_test.py

- Ricordati che devi fare vedere i vari logs che ottieni, magari mostrando anche qualche screenshot che hai ottenuto. 
- Ricorda che devi mostrare tutti i comandi che hai fatto per creare il cluster e tutti i comandi che vengono utilizzati da terminale per runnare i jobs, come: kubectl apply -f raycluster.yaml, kubectl apply -f rayjob.yaml, kubectl get pods, kubectl logs <pod_name>, ecc...


