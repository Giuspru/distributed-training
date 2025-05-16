# Introduzione:
Il progetto "Distributed-Training" ha l’obiettivo di seguire tutti i passaggi necessari per addestrare un modello di rete neurale, distribuendo il processo di training su un cluster Kubernetes composto da più nodi.
Nel progetto sono implementati tutti gli step fondamentali per distribuire un processo di machine learning su un cluster, partendo dalla configurazione infrastrutturale del cluster Kubernetes e del cluster Ray, passando per l’implementazione delle comunicazioni tra il cluster e un server MinIO esterno, fino ad arrivare alla fase di addestramento e validazione del modello di machine learning.

L’infrastruttura del progetto è composta da:
 - Un cluster Kubernetes formato da 3 nodi: un nodo master e due nodi worker.
 - Un cluster Ray con 8 worker distribuiti sui due nodi worker del cluster Kubernetes.
 - Un server MinIO, responsabile della gestione del dataset MNIST e del modello addestrato.

 ![Logo del progetto](./img/img.png) 



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


