pipeline {
    agent any
    stages {
        stage('Test run') {
            steps {
                echo "Test Run ..."
            }
        }

//         stage('Build Docker Image') {
//             steps {
//                 echo "Building Docker image..."
//                 sh 'docker build -t $DOCKER_IMAGE .'
//             }
//         }
//
//         stage('Push Docker Image') {
//             steps {
//                 echo "Pushing Docker image to registry..."
//                 withCredentials([usernamePassword(credentialsId: 'dockerhub-credentials-id', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
//                     sh '''
//                         echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin
//                         docker push $DOCKER_IMAGE
//                     '''
//                 }
//             }
//         }

//         stage('Deploy to Kubernetes') {
//             steps {
//                 echo "Deploying to Kubernetes cluster..."
//                 withCredentials([file(credentialsId: 'kubeconfig-credentials-id', variable: 'KUBECONFIG')]) {
//                     sh '''
//                         export KUBECONFIG=$KUBECONFIG
//                         kubectl apply -f k8s/k8s-deployment.yaml
//                     '''
//                 }
//             }
//         }
//     }

    post {
        success {
            echo "✅ Pipeline executed successfully."
        }
        failure {
            echo "❌ Pipeline failed!"
        }
    }
}
