

# raycluster.yaml:

Questo file raycluster.yaml è un manifest Kubernetes che definisce la configurazione di un Ray Cluster distribuito per l’esecuzione di carichi di lavoro in ambiente containerizzato, come il training distribuito di modelli di machine learning. In particolare, è pensato per essere utilizzato nel contesto di un progetto di training basato su Ray e MinIO.

Di seguito cosa fa nello specifico il manifest: 

Crea un RayCluster nel cluster Kubernetes con:
1 nodo head (capo-cluster), responsabile della gestione del cluster Ray.
1 nodo worker (lavoratore), che esegue i task distribuiti assegnati dal nodo head.
Versione di Ray utilizzata: 2.6.0.
Immagine Docker: viene utilizzata l’immagine personalizzata biancoj/peppe-train:0.1.0, che presumibilmente contiene il codice applicativo per il training o inferenza.
Risorse allocate:
Ogni nodo (head e worker) ha un limite di risorse pari a 1 CPU e 1 GiB di memoria.
Accesso ai segreti:
Entrambi i nodi montano le credenziali da un Secret Kubernetes chiamato minio-credentials. Questo è utile, ad esempio, per accedere a uno storage MinIO che contiene dataset o modelli.
Configurazione head node:
Espone il dashboard Ray su 0.0.0.0 per il monitoraggio dell’attività del cluster.
Service Account:
Il nodo head utilizza un ServiceAccount chiamato ray, il quale può essere configurato con i permessi necessari per accedere ad altri servizi nel cluster.

# rayjob.yaml
Questo file è un manifest Kubernetes che definisce un oggetto di tipo RayJob, utilizzato per eseguire automaticamente un job Ray all’interno di un cluster Ray già attivo. In questo caso specifico, il job avvia un training sul dataset MNIST utilizzando uno script Python presente nell'immagine Docker.

Di seguito cosa fa nello specifico il manifest:
Avvia un RayJob nel cluster Kubernetes con le seguenti caratteristiche:
Utilizza il cluster Ray denominato mnist-raycluster (specificato tramite clusterSelector).
Lancia lo script Python /app/train_mnist_ray.py con i seguenti argomenti:
--epochs 5: esegue il training per 5 epoche.
--batch-size 64: utilizza batch da 64 campioni.
Immagine Docker utilizzata: biancoj/peppe-train:0.1.0, che deve contenere lo script train_mnist_ray.py e tutte le dipendenze necessarie (es. Ray, PyTorch, ecc.).
Pod sottomettitore (submitter pod):
È il pod che lancia il job Ray e monitora l’esecuzione.
Non viene riavviato in caso di fallimento (restartPolicy: Never).
Contiene un solo container chiamato submitter.
Monta le credenziali da un Secret Kubernetes chiamato minio-credentials per l'accesso, ad esempio, a dati o modelli in uno storage esterno (come MinIO).

# service-account.yaml

Service Account Manifest - service-account.yaml
Questo file service-account.yaml definisce una ServiceAccount Kubernetes chiamata ray, creata nel namespace default. Si tratta di un componente fondamentale per consentire a pod specifici (in questo caso, i nodi del cluster Ray) di interagire in modo sicuro con l’API server di Kubernetes e accedere a risorse del cluster in base ai permessi assegnati.
Di seguito cosa fa nello specifico il manifest:
Crea una ServiceAccount con:
Nome: ray
Namespace: default (lo spazio dei nomi di default in Kubernetes)
Questa ServiceAccount può essere associata a pod (come nel nodo head del RayCluster) per gestire permessi di accesso alle risorse Kubernetes.