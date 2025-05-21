 # Implementazione Infrastrutturale: 

 ## 1. Cluster Kubernetes:
 In prima istanza sono state create tre macchine virtuali su di un Server privato, con i seguenti hostname:
 - **masternode** (192.168.122.33) con: 10 CPUs & 22Gb di RAM
 - **workernode1** (192.168.122.22) con: 8 CPUs & 17Gb di RAM
 - **workernode2** (192.168.122.94) con: 8 CPUs & 17Gb di RAM

Il masternode svolgerà il ruolo di control plane del cluster Kubernetes, mentre i due nodi worker1 e worker2 ricopriranno il ruolo di workersnode.

### 1.1 Installazione di Kubernetes con rke2 & avvio del servizio:
- Per installare Kubernetes sul cluster di macchine virtuali è stato utilizzato il tool **rke2**, dando il comando:

<pre lang="markdown">

curl -sfL https://get.rke2.io/ | INSTALL_RKE2_TYPE=server sh -

</pre>

In questo contesto, **RKE2** è stato scelto per semplificare l’installazione e la gestione del cluster Kubernetes sulle macchine virtuali, infatti questa tecnologia permette di include automaticamente componenti fondamentali come containerd, kube-proxy, kubelet, etcd e cni, fondamentali per la gestione del cluster.
Nello specifico, l’uso del comando sopra consente di installare il nodo server principale del cluster, avviando automaticamente i servizi necessari per eseguire Kubernetes.



- Per abilitare il servizio utilizzando rke2, e successivamente metterlo in atto vengono dati i due comandi: 

<pre lang="markdown">

 systemctl enable rke2-server.service
 systemctl start rke2-server.service

 </pre>



- Una volta completata l’installazione del servizio, il file di configurazione kubeconfig viene generato automaticamente e salvato nel percorso: 
<pre lang="markdown">

/etc/rancher/rke2/rke2.yaml

</pre>

Il kubeconfig è un file fondamentale che contiene tutte le informazioni necessarie per permettere al client kubectl di interagire con il cluster Kubernetes.
Per consentire a kubectl di localizzare correttamente questo file senza doverne specificare il percorso ogni volta, è necessario esportare la variabile d’ambiente KUBECONFIG con il seguente comando:

<pre lang="markdown">

export KUBECONFIG=/etc/rancher/rke2/rke2.yaml

</pre>

In questo modo, kubectl utilizzerà automaticamente il file di configurazione specificato per eseguire comandi verso il cluster. Ciò consente di semplificare l’interazione con Kubernetes, passando da un comando esplicito come:

<pre lang="markdown">

/var/lib/rancher/rke2/bin/kubectl get nodes

</pre>

a un comando più semplice e diretto come:

<pre lang="markdown">

kubectl get nodes

</pre>

In queto caso l'output ottenuto è il seguente: [immagine da mettere](./img/img3.png)
 ``` NAME STATUS ROLES AGE VERSION masternode0.localdomain Ready control-plane,etcd,master 17m v1.31.8+rke2r1 ```

Il Masternode è stato creato correttamente ed è pronto all'uso. Questo lo si evince dallo stato "ready" riportato nell'immagine sopra.

Mediante il comando: 

<pre lang="markdown">

kubectl get pods -A

</pre>

vengono mostrati tutti i pod in esecuzione nel cluster: [immagine da mettere](./img/img3.png)

### 1.2 Installazione di rke2 sui nodi worker:

A questo punto è necessario procedere con l’installazione di RKE2 anche sui nodi worker del cluster. Questo passaggio è fondamentale per consentire a tali nodi di unirsi correttamente al cluster e iniziare a eseguire i carichi di lavoro (i pod) orchestrati dal piano di controllo.

L’installazione sui nodi worker si effettua utilizzando il comando:

<pre lang="markdown">

curl -sfL https://get.rke2.io/ | INSTALL_RKE2_TYPE=agent sh -

</pre>

In questo caso, la variabile d’ambiente **INSTALL_RKE2_TYPE=agent** indica che il nodo verrà configurato come **agente**, ovvero come **worker node**. A differenza del masternode, i worker non eseguono i componenti del control plane, ma si occupano esclusivamente di eseguire i container e le applicazioni distribuite dal cluster.

Dopo l’installazione del componente agent, è necessario creare la directory di configurazione per RKE2:

<pre lang="markdown">

sudo mkdir -p /etc/rancher/rke2

</pre>

All’interno di questa directory verrà creato il file config.yaml, che conterrà i parametri essenziali per consentire al nodo worker di comunicare con il server del cluster, come ad esempio l’indirizzo del server API e il token di registrazione. La corretta configurazione di questo file è indispensabile per completare l’unione del nodo al cluster in modo sicuro ed efficace.


Il seguente comando consente di creare e scrivere direttamente il file config.yaml con i parametri richiesti:

<pre lang="markdown">

sudo tee /etc/rancher/rke2/config.yaml 

</pre>

Questo comando crea il file di configurazione **config.yaml** che consente al nodo worker di:
    - sapere a quale nodo server connettersi per unirsi al cluster RKE2;
    - autenticarsi in modo sicuro utilizzando un token generato dal masternode.
È un passaggio cruciale per completare correttamente l’aggiunta del nodo worker al cluster Kubernetes gestito con RKE2.


Una volta creato il file di configurazione config.yaml, è necessario abilitare e avviare il servizio RKE2 in modalità agent sul nodo worker. Questo viene fatto tramite i seguenti comandi:

<pre lang="markdown">

sudo systemctl enable rke2-agent.service
sudo systemctl start rke2-agent.service

</pre>

- Il comando **enable** assicura che il servizio venga avviato automaticamente ad ogni riavvio del sistema.
- Il comando **start** avvia immediatamente il servizio, consentendo al nodo di tentare la connessione al control plane.


Dopo aver avviato con successo il servizio rke2-agent, è possibile tornare sul nodo control plane e utilizzare kubectl per verificare lo stato dei nodi che compongono il cluster:

<pre lang="markdown">

kubectl get nodes

</pre>

A questo punto, si dovrebbe visualizzare un output simile al seguente: [immagine da mettere](./img/img4.png)

Entrambi i nodi risultano nello stato "Ready", il che indica:
- la comunicazione tra i nodi è avvenuta correttamente,
- i componenti fondamentali del nodo worker (kubelet, container runtime, plugin di rete, ecc.) sono operativi,
- il cluster è in grado di schedulare e gestire i carichi di lavoro in modo distribuito.

Questo risultato non è affatto banale, poiché implica che la configurazione della rete tra i nodi è stata effettuata con successo. Uno degli aspetti più delicati in un cluster Kubernetes è infatti proprio la corretta configurazione del networking pod-to-pod e node-to-node.


### 1.2 Installazione di Helm:
Sul nodo control plane del cluster k8s installiamo helm (da dare una letta su cosa e')

 <pre lang="markdown">curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh

</pre>

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

Si è scaricato il dataset MNIST localmente (nel master?) con i comandi: 

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


