import boto3 # Libreria python che comunica con object storage s3: Minio crea oggetti bucketstorage che sono compatibili con protoocollo s3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from botocore.config import Config

class S3BucketHandler:

    #init della classe, prende in ingresso un URL e il nome del bucket
    def __init__(self, endpoint_url, bucket_name):
        self.bucket_name = bucket_name
        self.access_key = "key" # <-- inserire la key corretta per accedere all'endpoint
        self.secret_key = "secret" # <-- inserire la secret key corretta per accedere all'endpoint
        
        #Creazione del client s3 
        try:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
            )
        except (NoCredentialsError, PartialCredentialsError) as e:
            print("Error: Invalid AWS credentials.", e)

    # Metodo per ottenere tutti i file presenti nel bucket.
    def get_all_files(self):
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            if 'Contents' in response:
                # Sort the objects by LastModified in descending order
                sorted_objects = sorted(response['Contents'], key=lambda obj: obj['LastModified'], reverse=True)
                
                # Build a list of dictionaries with file name and last modified timestamp
                files_info = [
                    {
                        'file_name': obj['Key'],
                        'last_modified': obj['LastModified']
                    }
                    for obj in sorted_objects
                ]
                return files_info
            else:
                return []
        except Exception as e:
            print("Error retrieving files:", e)
            return []


    # Metodo per caricare dei files nel bucket.
    def upload_file(self, file_path, object_name=None):
        try:
            object_name = object_name or file_path.split('/')[-1]
            self.s3_client.upload_file(file_path, self.bucket_name, object_name)
            print(f"File {file_path} uploaded successfully as {object_name}.")
        except Exception as e:
            print("Error uploading file:", e)

    # Metodo per scaricare files dal bucket.
    def download_file(self, object_name, file_path):
        try:
            print("Downloading file..., object_name: ", object_name)
            print("file_path: ", file_path)
            self.s3_client.download_file(self.bucket_name, object_name, file_path)
            print(f"File {object_name} downloaded successfully to {file_path}.")
        except Exception as e:
            print("Error downloading file:", e)

    # Metodo per cancellare files dal bucket.
    def delete_file(self, object_name):
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=object_name)
            print(f"File {object_name} deleted successfully.")
        except Exception as e:
            print("Error deleting file:", e)

    # Metodo per ottenere le informazioni su un file specifico. Nome del Bucket e nome del file.
    def get_file_info(self, object_name):
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=object_name)
            return response
        except Exception as e:
            print("Error retrieving file info:", e)


    # Metodo che restituisce la dimensione del bucket.
    def get_bucket_size(self):
        try:
            total_size = 0
            continuation_token = None
            
            while True:
                list_params = {
                    'Bucket': self.bucket_name,
                    'MaxKeys': 1000
                }
                
                if continuation_token:
                    list_params['ContinuationToken'] = continuation_token
                
                response = self.s3_client.list_objects_v2(**list_params)
                
                cnt = 0
                if 'Contents' in response:
                    for obj in response['Contents']:
                        cnt = cnt + 1
                        total_size += obj['Size']
                
                print("cnt", cnt)
                # If the response is truncated, we need to get the next page
                if response.get('IsTruncated'):
                    continuation_token = response['NextContinuationToken']
                else:
                    break
            
            return total_size
        
        except Exception as e:
            print("Error calculating bucket size:", e)
            return 0


    # Metodo che cancella tutti i file presenti nel bucket.
    def clean_bucket(self):
        try:
            files = self.get_all_files()
            if not files:
                print("Bucket is already empty.")
                return

            confirmation = input(f"Are you sure you want to delete all {len(files)} files in the bucket? (yes/no): ").strip().lower()
            if confirmation != 'yes':
                print("Operation canceled.")
                return

            for file in files:
                self.delete_file(file)

            print("All files deleted successfully.")
        except Exception as e:
            print("Error cleaning bucket:", e)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Interact with an S3 bucket.")
    parser.add_argument('--endpoint', required=True, help="S3 endpoint URL.")
    parser.add_argument('--bucket', required=True, help="Bucket name.")
    parser.add_argument('operation', choices=['get', 'download', 'upload', 'delete', 'get-info', 'get-size', 'clean'], help="Operation to perform.")
    parser.add_argument('--file', help="File path for upload or file name for delete/get-info.")

    args = parser.parse_args()

    handler = S3BucketHandler(args.endpoint, args.bucket)

    if args.operation == 'get':
        files = handler.get_all_files()
        if files:
            print("Files in bucket:", files)
            print(f"Total files: {len(files)}")
        else:
            print("No files found in bucket.")

    elif args.operation == 'upload':
        if not args.file:
            print("Error: --file argument is required for upload.")
        else:
            handler.upload_file(args.file)

    elif args.operation == 'delete':
        if not args.file:
            print("Error: --file argument is required for delete.")
        else:
            handler.delete_file(args.file)

    elif args.operation == 'get-info':
        if not args.file:
            print("Error: --file argument is required for get-info.")
        else:
            info = handler.get_file_info(args.file)
            if info:
                print("File info:", info)

    elif args.operation == 'download':
        if not args.file:
            print("Error: --file argument is required for download.")
        else:
            print("Going to download file: ", args.file)
            handler.download_file(args.file, args.file)

    elif args.operation == 'clean':
        handler.clean_bucket()

    elif args.operation == 'get-size':
        size = handler.get_bucket_size()
        if size is not None:
            print(f"Total bucket size: {size} bytes.")
            # return size in KB
            print(f"Total bucket size: {size/1024} KB.")
            # return size in MB
            print(f"Total bucket size: {size/1024/1024} MB.")
            # return size in GB
            print(f"Total bucket size: {size/1024/1024/1024} GB.")