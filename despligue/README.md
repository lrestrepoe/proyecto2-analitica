# Despliegue — Dashboard Saber 11 Bolívar

## URL del tablero
http://54.166.58.225:8050

## Descripción
El tablero fue desplegado en AWS usando contenedores Docker con ECS Fargate,
lo que permite que corra permanentemente en la nube sin depender de ninguna 
máquina local.

## Infraestructura utilizada
- **EC2**: t2.medium, Ubuntu 24.04 — usada para construir y subir la imagen
- **ECR**: Registro privado donde se almacena la imagen Docker
- **ECS + Fargate**: Servicio que corre el contenedor permanentemente

## Requisitos para replicar el despliegue
- Cuenta de AWS con acceso a EC2, ECR y ECS
- Docker instalado en la máquina
- Credenciales AWS configuradas (aws configure)

## Pasos para replicar
1. Construir la imagen Docker desde la raíz del proyecto:
docker build -t saber11:latest .

2. Autenticarse en ECR:
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <URI_ECR>

3. Etiquetar y subir la imagen:
docker tag saber11:latest <URI_ECR>:latest
docker push <URI_ECR>:latest

4. Crear cluster y servicio en ECS con Fargate apuntando a la imagen en ECR
5. Abrir el puerto 8050 en el security group del servicio

## Reiniciar el servicio
Si el servicio está detenido, ir a:
ECS → Clusters → saber11_cluster → Services → saber11-service → Update → Desired tasks: 1

**Nota:** La IP pública cambia cada vez que se reinicia el servicio.
Ir a Tasks → click en la tarea → Configuration → Public IP para obtener la nueva IP.