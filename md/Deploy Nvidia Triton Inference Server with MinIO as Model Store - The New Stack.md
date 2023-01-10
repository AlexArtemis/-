> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [thenewstack.io](https://thenewstack.io/deploy-nvidia-triton-inference-server-with-minio-as-model-store/)

> This tutorial shows how to set up the Nvidia Triton Inference Server that treats the MinIO tenant as ......

This tutorial is the latest part of a series where we build an end-to-end stack to perform machine learning inference at the edge. In the previous part of [this tutorial series](https://www.thenewstack.io/tutorial-installing-and-configuring-minio-as-a-model-registry-on-rke2), we installed the [MinIO](https://min.io/?utm_content=inline-mention) object storage service on SUSE Rancher’s RKE2 Kubernetes distribution. We will extend that use case further by deploying [Nvidia Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server) that treats the MinIO tenant as a model store.

![](https://cdn.thenewstack.io/media/2021/11/ec07c125-nvidia-triton-inference-server-dynamic-scalability-1024x576.jpeg)

By the end of this tutorial, we will have a fully configured model server and registry ready for inference.

Step 1 — Populate the MinIO Model Store with Sample Models
----------------------------------------------------------

Before deploying the model server, we need to have the model store or repository populated with a few models.

Start by cloning the Triton Inference Server GitHub repository.

`git clone https://github.com/triton-inference-server/server.git`

We will now run a shell script to download the models to the local filesystem, after which we will upload them to a MinIO bucket.

Run the `./fetch_models.sh` script available at `server/docs/examples` directory.

Wait for all the models to get downloaded in the `model_repository` directory. It may take a few minutes, depending on your Internet connection.

![](https://cdn.thenewstack.io/media/2021/11/70a40d11-triton-rke-0-1024x536.png)

Let’s use the MinIO CLI to upload the models from the `model_repository` directory to the `models` bucket. The bucket was created within the model-registry tenant created in the last tutorial.

Run the command from the `model_repository` directory to copy the files to the bucket.

`mc --insecure cp --recursive . model-registry/models`

Check the uploads by visiting the MinIO Console. You should be able to see the directories copied to the `models` bucket.

![](https://cdn.thenewstack.io/media/2021/11/47654e43-triton-rke-1-1024x675.png)

We are now ready to point NVIDIA Triton Inference Server to MinIO.

Step 2 — Deploy Triton Inference Server on RKE2
-----------------------------------------------

Triton expects Amazon S3 as the model store. To access the bucket, it needs a secret with the AWS credentials.

In our case, these credentials are essentially the MinIO tenant credentials saved from the last tutorial.  
Create a namespace and the secret within that.

`kubectl create ns model-server`

`kubectl create secret generic aws-credentials --from-literal=AWS_ACCESS_KEY_ID=admin --from-literal=AWS_SECRET_ACCESS_KEY=7c5c084d-9e8e-477b-9a2c-52bbf22db9af -n model-server`

Don’t forget to replace the credentials with your values.

Now, create the deployment, service and apply them.
# Create Triton deployment
cat <<EOF > triton-deploy.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: triton
  name: triton
  namespace: model-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: triton
  template:
    metadata:
      labels:
        app: triton
    spec:
      containers:
      - image: nvcr.io/nvidia/tritonserver:21.09-py3
        name: tritonserver
        command: ["/bin/bash"]
        args: ["-c", "cp /var/run/secrets/kubernetes.io/serviceaccount/ca.crt /usr/local/share/ca-certificates && update-ca-certificates && /opt/tritonserver/bin/tritonserver --model-store=s3://https://minio.model-registry.svc.cluster.local:443/models/ --strict-model-config=false"]
        env:
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: aws-credentials
              key: AWS_ACCESS_KEY_ID
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: aws-credentials
              key: AWS_SECRET_ACCESS_KEY      
        ports:
          - containerPort: 8000
            name: http
          - containerPort: 8001
            name: grpc
          - containerPort: 8002
            name: metrics
        volumeMounts:
        - mountPath: /dev/shm
          name: dshm
      volumes:
      - name: dshm
        emptyDir:
          medium: Memory
EOF
<table data-hpc="" data-tab-size="8" data-paste-markdown-skip="" data-tagsearch-lang="Shell" data-tagsearch-path="triton-minio.sh"><tbody><tr><td data-line-number="1"></td><td># Create Triton deployment</td></tr><tr><td data-line-number="2"></td><td>cat &lt;&lt;EOF &gt; triton-deploy.yaml</td></tr><tr><td data-line-number="3"></td><td>apiVersion: apps/v1</td></tr><tr><td data-line-number="4"></td><td>kind: Deployment</td></tr><tr><td data-line-number="5"></td><td>metadata:</td></tr><tr><td data-line-number="6"></td><td>labels:</td></tr><tr><td data-line-number="7"></td><td>app: triton</td></tr><tr><td data-line-number="8"></td><td>name: triton</td></tr><tr><td data-line-number="9"></td><td>namespace: model-server</td></tr><tr><td data-line-number="10"></td><td>spec:</td></tr><tr><td data-line-number="11"></td><td>replicas: 1</td></tr><tr><td data-line-number="12"></td><td>selector:</td></tr><tr><td data-line-number="13"></td><td>matchLabels:</td></tr><tr><td data-line-number="14"></td><td>app: triton</td></tr><tr><td data-line-number="15"></td><td>template:</td></tr><tr><td data-line-number="16"></td><td>metadata:</td></tr><tr><td data-line-number="17"></td><td>labels:</td></tr><tr><td data-line-number="18"></td><td>app: triton</td></tr><tr><td data-line-number="19"></td><td>spec:</td></tr><tr><td data-line-number="20"></td><td>containers:</td></tr><tr><td data-line-number="21"></td><td>- image: nvcr.io/nvidia/tritonserver:21.09-py3</td></tr><tr><td data-line-number="22"></td><td>name: tritonserver</td></tr><tr><td data-line-number="23"></td><td>command: ["/bin/bash"]</td></tr><tr><td data-line-number="24"></td><td>args: ["-c", "cp /var/run/secrets/kubernetes.io/serviceaccount/ca.crt /usr/local/share/ca-certificates &amp;&amp; update-ca-certificates &amp;&amp; /opt/tritonserver/bin/tritonserver --model-store=s3://https://minio.model-registry.svc.cluster.local:443/models/ --strict-model-config=false"]</td></tr><tr><td data-line-number="25"></td><td>env:</td></tr><tr><td data-line-number="26"></td><td>- name: AWS_ACCESS_KEY_ID</td></tr><tr><td data-line-number="27"></td><td>valueFrom:</td></tr><tr><td data-line-number="28"></td><td>secretKeyRef:</td></tr><tr><td data-line-number="29"></td><td>name: aws-credentials</td></tr><tr><td data-line-number="30"></td><td>key: AWS_ACCESS_KEY_ID</td></tr><tr><td data-line-number="31"></td><td>- name: AWS_SECRET_ACCESS_KEY</td></tr><tr><td data-line-number="32"></td><td>valueFrom:</td></tr><tr><td data-line-number="33"></td><td>secretKeyRef:</td></tr><tr><td data-line-number="34"></td><td>name: aws-credentials</td></tr><tr><td data-line-number="35"></td><td>key: AWS_SECRET_ACCESS_KEY</td></tr><tr><td data-line-number="36"></td><td>ports:</td></tr><tr><td data-line-number="37"></td><td>- containerPort: 8000</td></tr><tr><td data-line-number="38"></td><td>name: http</td></tr><tr><td data-line-number="39"></td><td>- containerPort: 8001</td></tr><tr><td data-line-number="40"></td><td>name: grpc</td></tr><tr><td data-line-number="41"></td><td>- containerPort: 8002</td></tr><tr><td data-line-number="42"></td><td>name: metrics</td></tr><tr><td data-line-number="43"></td><td>volumeMounts:</td></tr><tr><td data-line-number="44"></td><td>- mountPath: /dev/shm</td></tr><tr><td data-line-number="45"></td><td>name: dshm</td></tr><tr><td data-line-number="46"></td><td>volumes:</td></tr><tr><td data-line-number="47"></td><td>- name: dshm</td></tr><tr><td data-line-number="48"></td><td>emptyDir:</td></tr><tr><td data-line-number="49"></td><td>medium: Memory</td></tr><tr><td data-line-number="50"></td><td>EOF</td></tr></tbody></table><table data-hpc="" data-tab-size="8" data-paste-markdown-skip="" data-tagsearch-lang="Shell" data-tagsearch-path="triton-service.sh"><tbody><tr><td data-line-number="1"></td><td># Create Triton service</td></tr><tr><td data-line-number="2"></td><td>cat &lt;&lt;EOF &gt; triton-service.yaml</td></tr><tr><td data-line-number="3"></td><td>apiVersion: v1</td></tr><tr><td data-line-number="4"></td><td>kind: Service</td></tr><tr><td data-line-number="5"></td><td>metadata:</td></tr><tr><td data-line-number="6"></td><td>name: triton</td></tr><tr><td data-line-number="7"></td><td>namespace: model-server</td></tr><tr><td data-line-number="8"></td><td>spec:</td></tr><tr><td data-line-number="9"></td><td>type: NodePort</td></tr><tr><td data-line-number="10"></td><td>selector:</td></tr><tr><td data-line-number="11"></td><td>app: triton</td></tr><tr><td data-line-number="12"></td><td>ports:</td></tr><tr><td data-line-number="13"></td><td>- protocol: TCP</td></tr><tr><td data-line-number="14"></td><td>name: http</td></tr><tr><td data-line-number="15"></td><td>port: 8000</td></tr><tr><td data-line-number="16"></td><td>nodePort: 30800</td></tr><tr><td data-line-number="17"></td><td>targetPort: 8000</td></tr><tr><td data-line-number="18"></td><td>- protocol: TCP</td></tr><tr><td data-line-number="19"></td><td>name: grpc</td></tr><tr><td data-line-number="20"></td><td>port: 8001</td></tr><tr><td data-line-number="21"></td><td>nodePort: 30801</td></tr><tr><td data-line-number="22"></td><td>targetPort: 8001</td></tr><tr><td data-line-number="23"></td><td>- protocol: TCP</td></tr><tr><td data-line-number="24"></td><td>name: metrics</td></tr><tr><td data-line-number="25"></td><td>nodePort: 30802</td></tr><tr><td data-line-number="26"></td><td>port: 8002</td></tr><tr><td data-line-number="27"></td><td>targetPort: 8002</td></tr><tr><td data-line-number="28"></td><td>EOF</td></tr></tbody></table>

`kubectl apply -f triton-deploy.yaml`  
`kubectl apply -f triton-service.yaml`

![](https://cdn.thenewstack.io/media/2021/11/43283079-triton-rke-2-1024x293.png)

To make the Triton pod access Minio service, we fixed the certificate issue with the below command:

`cp /var/run/secrets/kubernetes.io/serviceaccount/ca.crt /usr/local/share/ca-certificates && update-ca-certificates`

We passed the MinIO bucket to Triton using the standard Amazon S3 convention – `s3://https://minio.model-registry.svc.cluster.local:443/models/`

Finally, check the logs of the Triton pod and make sure everything is working properly.

`kubectl logs triton-59994bb95c-7hgt7 -n model-server`

![](https://cdn.thenewstack.io/media/2021/11/137b78f5-triton-rke-3-1024x670.png)

If you see the above in the output, it means that Triton is able to download the models from the model store and serve them through the HTTP and gRPC endpoints.

Step 3 — Run Inference Client against Triton
--------------------------------------------

Start by cloning the repo to get the code for inference.

`cd https://github.com/triton-inference-server/client.git`

`cat <> requirements.txt  
cat requirements.txt  
pillow  
numpy  
attrdict  
tritonclient  
google-api-python-client  
grpcio  
geventhttpclient  
boto3  
EOF  
`

`pip3 install -r requirements.txt`

Navigate to the `client/src/python/examples` directory and execute the following command

`  
python3 image_client.py \  
-u TRITON_HTTP_ENDPOINT \  
-m inception_graphdef \  
-s INCEPTION \  
-x 1 \  
-c 1 \  
car.jpg  
`

Replace TRITON_HTTP_ENDPOINT with the host and nodeport of the Triton service. Send an image of a car and you should see the below output:

![](https://cdn.thenewstack.io/media/2021/11/8b998975-triton-rke-4-1024x211.png)

The client has invoked the Trinton inference endpoint with a request to load the inception model already available in the model store. Triton has performed the inference and printed the labels based on the classification.

Congratulations! You have successfully deployed and configured the model server backed by a model store running at the edge.

Group Created with Sketch.
