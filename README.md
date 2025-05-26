# Introduzione:
Il progetto **"Distributed-Training"** ha l’obiettivo di seguire tutti i passaggi necessari per addestrare un modello di rete neurale, **distribuendo il processo di training su un cluster Kubernetes composto da più nodi**.

Nel progetto sono implementati tutti gli steps fondamentali per distribuire un processo di machine learning su un cluster, partendo dalla configurazione infrastrutturale del cluster Kubernetes e del cluster Ray (**README.md** nella repo K8s) , passando per l’implementazione delle comunicazioni tra il cluster e un server **MinIO** esterno (**CONFIGMINIO.md** della repo dataset e file **README.md** della repo Utils), fino ad arrivare alla fase di addestramento e validazione del modello di machine learning (**README.md** della repo **Distributed_training_test**).

L’infrastruttura del progetto è composta da:
 - Un cluster Kubernetes formato da 3 nodi: un nodo master e due nodi worker.
 - Un cluster Ray con 8 worker distribuiti sui due nodi worker del cluster Kubernetes.
 - Un server MinIO, responsabile della gestione del dataset **MNIST** e del modello addestrato.

**In sintesi**:
Il modello di machine learning sarà addestrato utilizzando i dati presenti nel dataset MNIST, archiviati all'interno del bucket MINIO. Una volta completato l'addestramento, il modello risultante verrà scaricato in locale dal bucket. Successivamente, ne verrà valutata l’accuratezza impiegando i dati di test, anch’essi scaricati dal server MINIO.


 ![Logo del progetto](./img/img.png) 





